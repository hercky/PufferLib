/*
 * WEF-Cleanup unit checks (CPU only, no trainer, no GPU).
 *
 *   ./ocean/wef/run_test.sh
 *
 * Checks, against the real wef.h code paths:
 *   1. sensing: a waste item is observable through the mormyromast path within the
 *      sensing range, with the opposite sign to a pellet, and invisible beyond it;
 *   2. dynamics: waste inflow rate/cap, zero regrowth above the threshold, regrowth
 *      only inside the orchard;
 *   3. actions: a bite on waste in the cone removes it and spends the cooldown;
 *      an eater in the cone eats; freeze blocks motion and grants immunity;
 *   4. baseline mode (cleanup=0) still initialises and steps with every new key at
 *      its default.
 */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wef.h"

static int g_fail = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { g_fail++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } \
    else { printf("ok   "); printf(__VA_ARGS__); printf("\n"); } } while (0)

typedef struct {
    Env* env;
    obs_t* obs;
    float* act;
    float* rew;
    float* term;
} Harness;

static void kw_base(Dict* kw, int num_fish) {
    dict_set(kw, "num_agents", num_fish);
    dict_set(kw, "min_arena_width", 60);
    dict_set(kw, "max_arena_width", 60);
    dict_set(kw, "min_arena_height", 40);
    dict_set(kw, "max_arena_height", 40);
    dict_set(kw, "food_distribution", FOOD_UNIFORM);
    dict_set(kw, "num_food", 64);
    dict_set(kw, "patch_radius", 6);
    dict_set(kw, "patch_radius_std", 1.5);
    dict_set(kw, "patch_density", 0.001);
    dict_set(kw, "electric_field_radius", 15);
    dict_set(kw, "reflection_wall_range", 100);
    dict_set(kw, "episode_length", 1024);
}

// The scripts/cleanup.args preset.
static void kw_cleanup(Dict* kw) {
    dict_set(kw, "cleanup", 1);
    dict_set(kw, "strip_cm", 10);
    dict_set(kw, "orchard_cm", 20);
    dict_set(kw, "spawn_band", 1);
    dict_set(kw, "waste_max", 32);
    dict_set(kw, "waste_start", 16);
    dict_set(kw, "waste_spawn_p", 0.10);
    dict_set(kw, "regrow_p_max", 0.006);
    dict_set(kw, "food_start", 0);
    dict_set(kw, "clean_priority", 1);
    dict_set(kw, "proximity_shaping", 0);
    dict_set(kw, "obs_extra", 2);
    dict_set(kw, "size_min", 0.5);
    dict_set(kw, "size_max", 0.5);
}

// Harvest preset: the paper's 70x70 patchy arena with density-dependent regrowth.
static void kw_harvest(Dict* kw) {
    dict_set(kw, "min_arena_width", 70);
    dict_set(kw, "max_arena_width", 70);
    dict_set(kw, "min_arena_height", 70);
    dict_set(kw, "max_arena_height", 70);
    dict_set(kw, "food_distribution", FOOD_PATCHY);
    dict_set(kw, "regrow_mode", 1);
    dict_set(kw, "regrow_p_max", 0.02);
    dict_set(kw, "regrow_radius_cm", 4);
    dict_set(kw, "size_min", 0.5);
    dict_set(kw, "size_max", 0.5);
}

static Harness make(Dict* kw, int num_fish) {
    Harness h = {0};
    h.env = (Env*)calloc(1, sizeof(Env));
    h.env->rng = 7;
    puf_init(h.env, kw);
    h.obs = (obs_t*)calloc((size_t)num_fish * OBS_SIZE, sizeof(obs_t));
    h.act = (float*)calloc((size_t)num_fish * NUM_ATNS, sizeof(float));
    h.rew = (float*)calloc(num_fish, sizeof(float));
    h.term = (float*)calloc(num_fish, sizeof(float));
    for (int i = 0; i < num_fish; i++) {
        h.env->agents[i].observations = h.obs + i * OBS_SIZE;
        h.env->agents[i].actions = h.act + i * NUM_ATNS;
        h.env->agents[i].rewards = h.rew + i;
        h.env->agents[i].terminals = h.term + i;
    }
    puf_reset(h.env);
    return h;
}

static void set_action(Harness* h, int i, float move_raw, float turn_raw, float eod_raw, float bite_raw) {
    h->act[i * NUM_ATNS + 0] = move_raw;
    h->act[i * NUM_ATNS + 1] = turn_raw;
    h->act[i * NUM_ATNS + 2] = eod_raw;
    h->act[i * NUM_ATNS + 3] = bite_raw;
}

static void place_fish(Env* env, int i, float x, float y, float orientation) {
    env->fish[i].pos = (Vec2){x, y};
    env->fish[i].orientation = orientation;
    env->fish[i].emits_eod = true;
    env->fish[i].eat_cooldown = 0;
    env->fish[i].bite_cooldown = 0;
    env->fish[i].freeze = 0;
}

static void clear_objects(Env* env) {
    for (int i = 0; i < env->num_food; i++) {
        env->food[i] = (FishFood){0};
    }
    for (int i = 0; i < env->waste_max; i++) {
        env->waste[i] = (Waste){0};
    }
    env->food_active = 0;
    env->waste_active = 0;
}

// Max-magnitude mormyromast reading of fish 0 (signed).
static float morm_peak(Env* env) {
    obs_t* obs = env->agents[0].observations;
    float peak = 0.0f;
    for (int s = 0; s < NUM_MORMYROMASTS; s++) {
        if (fabsf(obs[s]) > fabsf(peak)) {
            peak = obs[s];
        }
    }
    return peak;
}

static void test_sensing(void) {
    printf("-- sensing\n");
    Dict kw = {0};
    kw_base(&kw, 1);
    kw_cleanup(&kw);
    Harness h = make(&kw, 1);
    Env* env = h.env;
    clear_objects(env);
    place_fish(env, 0, 5.0f, 20.0f, 0.0f);   // in the strip, facing +x

    // pellet 4 cm ahead: reference sign
    env->food[0] = (FishFood){.pos = {9.0f, 20.0f}, .orientation = 0.0f, .active = true};
    env->food_active = 1;
    compute_observations(env);
    float pellet = morm_peak(env);
    CHECK(fabsf(pellet) > 0.0f, "pellet at 4 cm is sensed (peak %.3f)", pellet);
    clear_objects(env);

    // waste 8 cm ahead (on axis)
    env->waste[0] = (Waste){.pos = {13.0f, 20.0f}, .active = true};
    env->waste_active = 1;
    compute_observations(env);
    float waste8 = morm_peak(env);
    CHECK(fabsf(waste8) > 0.0f, "waste at 8 cm on-axis is sensed (peak %.3f)", waste8);
    CHECK(waste8 * pellet < 0.0f, "waste (+1.0) has the opposite sign to a pellet (-0.5)");

    // waste 8 cm broadside
    env->waste[0].pos = (Vec2){5.0f, 28.0f};
    compute_observations(env);
    float side8 = morm_peak(env);
    CHECK(fabsf(side8) > 0.0f, "waste at 8 cm broadside is sensed (peak %.3f)", side8);

    // waste 12 cm ahead: outside waste_sense_range_cm (10) -> zero
    env->waste[0].pos = (Vec2){17.0f, 20.0f};
    compute_observations(env);
    float far = morm_peak(env);
    CHECK(far == 0.0f, "waste at 12 cm is not sensed (peak %.3f)", far);

    // not emitting: waste is invisible through the induced path
    env->waste[0].pos = (Vec2){13.0f, 20.0f};
    env->fish[0].emits_eod = false;
    compute_observations(env);
    float silent = morm_peak(env);
    CHECK(silent == 0.0f, "a silent fish does not see waste (peak %.3f)", silent);
    env->fish[0].emits_eod = true;

    // ampullary channel (obs 36..59) never sees waste: identical readings with and
    // without the item from the same RNG state (the channel reads the fish's own
    // intrinsic dipole plus noise, so it is not zero).
    unsigned int rng_saved = env->rng;
    compute_observations(env);
    float with_waste[NUM_AMPULLARY];
    memcpy(with_waste, env->agents[0].observations + NUM_MORMYROMASTS, sizeof(with_waste));
    env->waste[0].active = false;
    env->waste_active = 0;
    env->rng = rng_saved;
    compute_observations(env);
    bool same = memcmp(with_waste, env->agents[0].observations + NUM_MORMYROMASTS, sizeof(with_waste)) == 0;
    CHECK(same, "waste is invisible to the ampullary channel (readings identical with/without)");
    env->waste[0].active = true;
    env->waste_active = 1;

    // obs_extra = 2: normalized x
    float extra = env->agents[0].observations[103];
    CHECK(fabsf(extra - (2.0f * 5.0f / 60.0f - 1.0f)) < 1e-6f, "obs[103] = 2x/W - 1 (%.4f)", extra);
    dict_clear(&kw);
}

static void test_dynamics(void) {
    printf("-- dynamics\n");
    Dict kw = {0};
    kw_base(&kw, 4);
    kw_cleanup(&kw);
    Harness h = make(&kw, 4);
    Env* env = h.env;
    CHECK(env->waste_active == 16 && env->food_active == 0, "reset: 16 waste, 0 pellets");
    for (int i = 0; i < env->waste_max; i++) {
        if (env->waste[i].active) {
            CHECK(env->waste[i].pos.x <= 10.0f, "waste %d inside the strip (x=%.1f)", i, env->waste[i].pos.x);
            break;
        }
    }
    // fish sit still (move raw -8), emit, no bite
    for (int i = 0; i < 4; i++) {
        set_action(&h, i, -8.0f, 0.0f, 1.0f, -1.0f);
    }
    for (int t = 0; t < 200; t++) {
        puf_step(env);
    }
    CHECK(env->waste_active > 16 && env->waste_active <= 32,
        "inflow after 200 steps: %d items (expect ~16 + 0.1*200 capped at 32)", env->waste_active);
    CHECK(env->regrown == 0, "no regrowth while waste >= theta*max (regrown %d)", env->regrown);
    CHECK(env->open_steps == 0, "orchard never opened (open_steps %d)", env->open_steps);

    // clean the strip by hand: 4 items left -> q = 1 - 4/12.8 = 0.69
    for (int i = 0, kept = 0; i < env->waste_max; i++) {
        if (env->waste[i].active && kept < 4) {
            kept++;
        } else {
            env->waste[i].active = false;
        }
    }
    env->waste_active = 4;
    int spawn_p = env->waste_spawn_delay;
    (void)spawn_p;
    env->waste_spawn_p = 0.0f;  // freeze inflow for this check
    int before = env->regrown;
    for (int t = 0; t < 200; t++) {
        puf_step(env);
    }
    int grown = env->regrown - before;
    CHECK(grown > 20 && grown < 120, "regrowth over 200 steps at q=0.69: %d pellets (expect ~0.26/step)", grown);
    bool all_in_orchard = true;
    for (int f = 0; f < env->num_food; f++) {
        if (env->food[f].active && env->food[f].pos.x < 40.0f) {
            all_in_orchard = false;
        }
    }
    CHECK(all_in_orchard, "every regrown pellet is inside the orchard (x >= 40)");
    CHECK(env->tick == 400 && env->episode == 0, "fixed-length episode: no early stop (tick %d)", env->tick);
    dict_clear(&kw);
}

static void test_actions(void) {
    printf("-- actions\n");
    Dict kw = {0};
    kw_base(&kw, 2);
    kw_cleanup(&kw);
    dict_set(&kw, "waste_spawn_p", 0);
    dict_set(&kw, "regrow_p_max", 0);
    dict_set(&kw, "bitten_freeze_steps", 25);
    Harness h = make(&kw, 2);
    Env* env = h.env;
    clear_objects(env);

    // fish 0 in the strip facing +x, waste 2 cm ahead, fish 1 far away
    place_fish(env, 0, 5.0f, 20.0f, 0.0f);
    place_fish(env, 1, 50.0f, 20.0f, 0.0f);
    env->waste[0] = (Waste){.pos = {7.0f, 20.0f}, .active = true};
    env->waste_active = 1;
    set_action(&h, 0, -8.0f, 0.0f, 1.0f, 1.0f);   // bite
    set_action(&h, 1, -8.0f, 0.0f, 1.0f, -1.0f);
    puf_step(env);
    CHECK(env->waste_active == 0 && env->cleans_by[0] == 1 && env->fish[0].cleaned == 1,
        "bite on waste in the cone removes it (waste_active %d, cleans_by %d)", env->waste_active, env->cleans_by[0]);
    CHECK(env->fish[0].bite_cooldown == 4, "cooldown spent (bite_cooldown %d after decrement)", env->fish[0].bite_cooldown);
    env->waste[0] = (Waste){.pos = {7.0f, 20.0f}, .active = true};
    env->waste_active = 1;
    puf_step(env);
    CHECK(env->waste_active == 1, "no second clean while cooling down");
    for (int t = 0; t < 4; t++) {
        puf_step(env);
    }
    CHECK(env->waste_active == 0, "clean works again after the cooldown");
    CHECK(env->fish[0].pos.x < 5.01f, "stationary fish did not move (x %.3f)", env->fish[0].pos.x);

    // eater: pellet 1.5 cm ahead of fish 1 -> eaten, reward +1
    env->food[0] = (FishFood){.pos = {51.5f, 20.0f}, .orientation = 0.0f, .active = true};
    env->food_active = 1;
    puf_step(env);
    CHECK(env->food_active == 0 && env->food_by[1] == 1 && h.rew[1] > 0.99f,
        "eater eats the pellet in its cone (reward %.3f)", h.rew[1]);

    // freeze: fish 0 bites fish 1 from 2 cm behind
    place_fish(env, 0, 48.0f, 20.0f, 0.0f);
    place_fish(env, 1, 50.0f, 20.0f, 0.0f);
    env->fish[0].bite_cooldown = 0;
    set_action(&h, 0, -8.0f, 0.0f, 1.0f, 1.0f);
    set_action(&h, 1, -8.0f, 0.0f, 1.0f, -1.0f);  // victim still (bites resolve after motion)
    puf_step(env);
    CHECK(env->fish[0].bite_victim == 1 && env->fish[1].freeze == 24 && h.rew[1] < -0.4f,
        "bite lands, victim frozen (freeze %d) and fined (%.3f)", env->fish[1].freeze, h.rew[1]);
    float x_before = env->fish[1].pos.x;
    set_action(&h, 1, 8.0f, 0.0f, 1.0f, -1.0f);   // now it tries to flee at full speed
    for (int t = 0; t < 10; t++) {
        env->fish[0].bite_cooldown = 0;
        puf_step(env);
    }
    CHECK(env->fish[1].pos.x == x_before, "frozen fish cannot move (x %.3f)", env->fish[1].pos.x);
    CHECK(env->bites == 1, "frozen fish is immune to further bites (bites %d)", env->bites);
    dict_clear(&kw);
}

static void test_harvest(void) {
    printf("-- harvest regrowth\n");
    Dict kw = {0};
    kw_base(&kw, 1);
    kw_harvest(&kw);
    Harness h = make(&kw, 1);
    Env* env = h.env;
    CHECK(env->cleanup == 0 && env->food_active == 64 && env->cur_episode_length == 1024,
        "harvest reset: 64 pellets, fixed-length episode");
    clear_objects(env);
    // isolated pellet eaten -> never regrows; a pellet with 3 active neighbours regrows in place
    env->food[0] = (FishFood){.pos = {10.0f, 10.0f}, .active = false};
    env->food[1] = (FishFood){.pos = {40.0f, 40.0f}, .active = false};
    env->food[2] = (FishFood){.pos = {41.0f, 40.0f}, .active = true};
    env->food[3] = (FishFood){.pos = {40.0f, 41.0f}, .active = true};
    env->food[4] = (FishFood){.pos = {42.0f, 42.0f}, .active = true};
    env->food_active = 3;
    place_fish(env, 0, 5.0f, 60.0f, 0.0f);
    set_action(&h, 0, -8.0f, 0.0f, 1.0f, -1.0f);
    CHECK(wef_food_neighbours(env, 1) == 3 && wef_food_neighbours(env, 0) == 0,
        "neighbour counts: dense slot 3, isolated slot 0");
    int t = 0;
    while (!env->food[1].active && t < 2000) {
        puf_step(env);
        t++;
    }
    CHECK(env->food[1].active && t < 2000, "dense slot regrew after %d steps (p = 0.02)", t);
    CHECK(env->food[1].pos.x == 40.0f && env->food[1].pos.y == 40.0f, "regrowth is at the slot's own position");
    CHECK(!env->food[0].active, "isolated slot never regrows");
    CHECK(env->episode == (t >= 1024 ? 1 : 0), "no early termination with regrowth on (episode %d)", env->episode);
    dict_clear(&kw);
}

static void test_baseline(void) {
    printf("-- baseline mode\n");
    Dict kw = {0};
    kw_base(&kw, 4);
    dict_set(&kw, "min_arena_width", 30);
    dict_set(&kw, "max_arena_width", 400);
    dict_set(&kw, "min_arena_height", 30);
    dict_set(&kw, "max_arena_height", 400);
    dict_set(&kw, "food_distribution", FOOD_RANDOM);
    dict_set(&kw, "episode_length", 512);
    Harness h = make(&kw, 4);
    Env* env = h.env;
    CHECK(env->cleanup == 0 && env->waste_max == 0 && env->proximity_shaping == PROXIMITY_SHAPING_REWARD
        && env->bitten_reward == BITTEN_REWARD && env->bite_reward == BITE_REWARD,
        "defaults reproduce baseline constants");
    CHECK(env->food_active == 64 && env->cur_episode_length == 512, "baseline reset: 64 pellets, 512-step episodes");
    for (int i = 0; i < 4; i++) {
        set_action(&h, i, 2.0f, 0.3f, 1.0f, 1.0f);
    }
    for (int t = 0; t < 600; t++) {
        puf_step(env);
    }
    CHECK(env->episode >= 1, "baseline episode terminates (episode %d)", env->episode);
    CHECK(env->agents[0].observations[103] == 0.0f, "obs[103] is the constant 0 in baseline mode");
    dict_clear(&kw);
}

// Calibration mode (E1): all fish scripted, so the whole thing runs on the CPU.
//   wef_test roles=3,3,2,2 [episodes=100] [oracle=0] [key=value ...]
// Prints per-slot pellets / cleans / strip time per episode and episode-level
// waste fraction and open fraction, in the scripts/cleanup.args preset.
static int run_calibration(int argc, char** argv) {
    Dict kw = {0};
    kw_base(&kw, 4);
    bool harvest = false;
    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "preset=harvest") == 0) {
            harvest = true;
        }
    }
    if (harvest) {
        kw_harvest(&kw);
    } else {
        kw_cleanup(&kw);
    }
    int episodes = 100;
    const char* roles = "0,0,0,0";
    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "preset=harvest") == 0) {
            continue;
        }
        char key[64];
        char val[64];
        if (sscanf(argv[a], "%63[^=]=%63s", key, val) != 2) {
            fprintf(stderr, "bad arg %s\n", argv[a]);
            return 2;
        }
        if (strcmp(key, "roles") == 0) {
            roles = argv[a] + 6;
        } else if (strcmp(key, "episodes") == 0) {
            episodes = atoi(val);
        } else if (strcmp(key, "oracle") == 0) {
            dict_set(&kw, "bot_oracle", atof(val));
        } else {
            dict_set(&kw, key, atof(val));
        }
    }
    puf_ini_set(&kw, "roles", roles);   // parses the comma list into values[]/len
    Harness h = make(&kw, 4);
    Env* env = h.env;
    double pellets[MAX_AGENTS] = {0};
    double cleans[MAX_AGENTS] = {0};
    double strip[MAX_AGENTS] = {0};
    double waste_frac = 0.0;
    double open_frac = 0.0;
    double collective = 0.0;
    double stock_left = 0.0;
    const char* dbg = getenv("WEF_TEST_DEBUG");
    for (int e = 0; e < episodes; e++) {
        while (env->episode == e) {
            puf_step(env);
            if (dbg && e == 0 && env->tick <= 80) {
                FishAgent* f = &env->fish[0];
                printf("t=%3d pos=(%.1f,%.1f) ori=%.2f move=%.2f turn=%.2f bite=%d eod=%d waste=%d\n",
                    env->tick, f->pos.x, f->pos.y, f->orientation, f->last_action[0], f->last_action[1],
                    f->bite_action, f->emits_eod, env->waste_active);
            }
            // log accumulates at episode end; read the per-episode counters before reset
            if (env->tick == env->cur_episode_length - 1) {
                for (int i = 0; i < env->num_agents; i++) {
                    pellets[i] += env->food_by[i];
                    cleans[i] += env->cleans_by[i];
                    strip[i] += (double)env->strip_steps_by[i] / env->cur_episode_length;
                }
                collective += env->food_eaten;
                stock_left += env->food_active;
                waste_frac += env->waste_frac_sum / env->cur_episode_length;
                open_frac += (double)env->open_steps / env->cur_episode_length;
            }
        }
    }
    printf("roles=%s oracle=%d episodes=%d\n", roles, env->bot_oracle, episodes);
    printf("  collective pellets/episode %.1f   waste_frac %.3f   open_frac %.3f   stock_left %.1f\n",
        collective / episodes, waste_frac / episodes, open_frac / episodes, stock_left / episodes);
    printf("  slot role pellets cleans strip%%  cleans/strip-step\n");
    for (int i = 0; i < 4; i++) {
        double strip_steps = strip[i] / episodes * env->episode_length;
        printf("  %4d %4d %7.1f %6.1f %5.1f%%  %.3f\n", i, env->roles[i], pellets[i] / episodes,
            cleans[i] / episodes, 100.0 * strip[i] / episodes,
            strip_steps > 0 ? cleans[i] / episodes / strip_steps : 0.0);
    }
    puf_close(env);
    dict_clear(&kw);
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1) {
        return run_calibration(argc, argv);
    }
    test_sensing();
    test_dynamics();
    test_actions();
    test_harvest();
    test_baseline();
    printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
