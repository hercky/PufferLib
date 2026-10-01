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
 *      its default;
 *   5. Allelopathic Harvest, docs/wef-allelopathic-harvest-v0-design.md 7.1, checks U1-U10
 *      (obs layout, ripening, eating, planting / hold / zaps, sensing, taste, normalisers,
 *      zap cooldown) plus the episode Log and the V3 trace. They run on both builds with
 *      min(8, MAX_AGENTS) fish; the slot numbers in the messages are derived from OBS_SIZE.
 *      U11 (default bench hash) and U12 (projection identity, `bench project=4`) are run
 *      by run_test.sh.
 */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

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

// Allelopathic Harvest, AH-strict 4+4 (docs/wef-allelopathic-harvest-v0-design.md 0.1b, 4.1):
// 8 fish, 40 x 40 cm, 64 uniform bushes, linear F, hold 32, 1024 steps, zaps off (learnability
// rungs; bitten_freeze_steps=25 on the command line is the zap arm). Needs the 8-fish build.
static void kw_allelo(Dict* kw) {
    dict_set(kw, "num_agents", 8);
    dict_set(kw, "min_arena_width", 40);
    dict_set(kw, "max_arena_width", 40);
    dict_set(kw, "min_arena_height", 40);
    dict_set(kw, "max_arena_height", 40);
    dict_set(kw, "food_distribution", FOOD_UNIFORM);
    dict_set(kw, "num_food", 64);
    dict_set(kw, "episode_length", 1024);
    dict_set(kw, "allelo", 1);
    dict_set(kw, "ripen_lin", 8.9e-3);
    dict_set(kw, "ripen_cubic", 0);
    dict_set(kw, "plant_steps", 32);
    dict_set(kw, "plant_mode", 1);
    dict_set(kw, "plant_split", 0.6745);
    dict_set(kw, "plant_bin_order", 0);
    dict_set(kw, "plant_priority", 0);
    dict_set(kw, "plant_radius_cm", 3.0);
    dict_set(kw, "taste_n_a", 4);
    dict_set(kw, "taste_match", 1.0);
    dict_set(kw, "taste_other", 0.5);
    dict_set(kw, "size_a_min", 0.58);
    dict_set(kw, "size_a_max", 0.62);
    dict_set(kw, "size_b_min", 0.38);
    dict_set(kw, "size_b_max", 0.42);
    dict_set(kw, "food_contrast_a", -0.5);
    dict_set(kw, "food_contrast_b", 0.5);
    dict_set(kw, "food_radius_unripe_cm", 0.12);
    dict_set(kw, "food_radius_ripe_cm", 0.30);
    dict_set(kw, "unripe_intrinsic", 0);
    dict_set(kw, "ripe_intrinsic", 3);
    dict_set(kw, "obs_extra", 5);
    dict_set(kw, "bitten_freeze_steps", 0);
    dict_set(kw, "zap_cooldown_steps", 25);
    dict_set(kw, "bitten_reward", 0);
    dict_set(kw, "bite_reward", 0);
    dict_set(kw, "proximity_shaping", 0);
}

// shape=conv: the AH-conv keys on top of kw_allelo (paper cubic F, hold 8, 2048 steps).
static void kw_allelo_conv(Dict* kw) {
    dict_set(kw, "ripen_lin", 1.7e-3);
    dict_set(kw, "ripen_cubic", 7.2e-3);
    dict_set(kw, "ripen_pow", 3);
    dict_set(kw, "plant_steps", 8);
    dict_set(kw, "episode_length", 2048);
}

static unsigned int g_seed = 7;  // calibration: seed=N on the command line

// Observation layout (section 3.1): 110 floats at MAX_AGENTS 4, 162 at 8.
#define OBS_META_START (NUM_MORMYROMASTS + NUM_AMPULLARY + NUM_KNOLLEN * (MAX_AGENTS - 1))
#define OBS_ACT_START (OBS_META_START + MAX_AGENTS - 1)   // last action (4 slots)
#define OBS_EXTRA (OBS_SIZE - 7)                           // obs_extra
#define OBS_BITTEN (OBS_SIZE - 6)
#define OBS_OWN_SIZE (OBS_SIZE - 5)
#define OBS_BITE_CD (OBS_SIZE - 4)
#define OBS_EAT_CD (OBS_SIZE - 1)                          // freeze ? 1 : eat_cooldown / max(3, plant_steps)

// stride: floats between consecutive obs rows (OBS_SIZE, or more to leave guard words).
static Harness make_ex(Dict* kw, int num_fish, int stride) {
    Harness h = {0};
    h.env = (Env*)calloc(1, sizeof(Env));
    h.env->rng = g_seed;
    puf_init(h.env, kw);
    h.obs = (obs_t*)calloc((size_t)num_fish * stride, sizeof(obs_t));
    h.act = (float*)calloc((size_t)num_fish * NUM_ATNS, sizeof(float));
    h.rew = (float*)calloc(num_fish, sizeof(float));
    h.term = (float*)calloc(num_fish, sizeof(float));
    for (int i = 0; i < num_fish; i++) {
        h.env->agents[i].observations = h.obs + i * stride;
        h.env->agents[i].actions = h.act + i * NUM_ATNS;
        h.env->agents[i].rewards = h.rew + i;
        h.env->agents[i].terminals = h.term + i;
    }
    puf_reset(h.env);
    return h;
}

static Harness make(Dict* kw, int num_fish) {
    return make_ex(kw, num_fish, OBS_SIZE);
}

static void destroy(Harness* h) {
    puf_close(h->env);
    free(h->env);
    free(h->obs);
    free(h->act);
    free(h->rew);
    free(h->term);
    *h = (Harness){0};
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
    float extra = env->agents[0].observations[OBS_SIZE - 7];
    CHECK(fabsf(extra - (2.0f * 5.0f / 60.0f - 1.0f)) < 1e-6f, "obs[extra] = 2x/W - 1 (%.4f)", extra);
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

static void test_commons(void) {
    printf("-- commons (global logistic stock) regrowth\n");
    Dict kw = {0};
    kw_base(&kw, 1);
    kw_harvest(&kw);
    dict_set(&kw, "regrow_mode", 2);
    dict_set(&kw, "regrow_p_max", 0.05);
    dict_set(&kw, "regrow_allee", 0.25);
    dict_set(&kw, "obs_extra", 3);
    Harness h = make(&kw, 1);
    Env* env = h.env;
    place_fish(env, 0, 5.0f, 60.0f, 0.0f);
    set_action(&h, 0, -8.0f, 0.0f, 1.0f, -1.0f);
    // At the critical stock A = 16 of K = 64 nothing regrows.
    for (int f = 0; f < 64; f++) {
        env->food[f].active = f < 16;
    }
    env->food_active = 16;
    for (int t = 0; t < 200; t++) {
        puf_step(env);
    }
    CHECK(env->food_active == 16, "no regrowth at the critical stock (S %d)", env->food_active);
    CHECK(fabsf(env->agents[0].observations[OBS_SIZE - 7] - 0.25f) < 1e-6f, "obs[extra] = S / K (%.3f)",
        env->agents[0].observations[OBS_SIZE - 7]);
    // Above it the stock grows toward K, with new pellets anywhere in the arena.
    env->food[16].active = true;
    env->food_active = 17;
    int t = 0;
    while (env->food_active < 48 && t < 2000) {
        puf_step(env);
        t++;
    }
    CHECK(env->food_active >= 48, "stock above A grows (S %d after %d steps)", env->food_active, t);
    dict_clear(&kw);
}

static void test_seasonal(void) {
    printf("-- seasonal (GovSim-style) regrowth and bite metrics\n");
    Dict kw = {0};
    kw_base(&kw, 2);
    kw_harvest(&kw);
    dict_set(&kw, "regrow_mode", 3);
    dict_set(&kw, "regrow_p_max", 0);
    dict_set(&kw, "season_steps", 50);
    dict_set(&kw, "season_growth", 2.0);
    dict_set(&kw, "regrow_allee", 0.1);
    Harness h = make(&kw, 2);
    Env* env = h.env;
    CHECK(env->regrows && env->cur_episode_length == 1024, "mode 3 counts as regrowth (fixed-length episode)");
    clear_objects(env);
    for (int f = 0; f < 20; f++) {
        env->food[f] = (FishFood){.pos = {60.0f + 0.1f * f, 60.0f}, .active = true};
    }
    env->food_active = 20;
    place_fish(env, 0, 5.0f, 5.0f, 0.0f);
    place_fish(env, 1, 10.0f, 5.0f, 0.0f);
    set_action(&h, 0, -8.0f, 0.0f, 1.0f, -1.0f);
    set_action(&h, 1, -8.0f, 0.0f, 1.0f, -1.0f);
    while (env->tick < 49) {
        puf_step(env);
    }
    CHECK(env->food_active == 20, "no growth within a season (S %d)", env->food_active);
    puf_step(env);
    CHECK(env->food_active == 40 && !env->collapsed, "season end doubles the stock (S %d)", env->food_active);
    // Drop the stock to the threshold (0.1 * 64 = 6.4): collapse at the next season end, for good.
    for (int f = 6; f < 64; f++) {
        env->food[f].active = false;
    }
    env->food_active = 6;
    while (env->tick < 150) {
        puf_step(env);
    }
    CHECK(env->collapsed && env->collapse_tick == 100 && env->food_active == 6,
        "stock at the threshold collapses and never regrows (collapsed %d at %d, S %d)",
        env->collapsed, env->collapse_tick, env->food_active);

    // Bite metrics: fish 1 ate while the stock was <= K/2, then fish 0 bites it.
    env->fish[1].eat_mark = env->tick;
    env->fish[1].low_eat_mark = env->tick;
    place_fish(env, 0, 20.0f, 20.0f, 0.0f);
    place_fish(env, 1, 22.0f, 20.0f, 0.0f);
    set_action(&h, 0, -8.0f, 0.0f, 1.0f, 1.0f);
    puf_step(env);
    CHECK(env->bites == 1 && env->bites_on_eaters == 1 && env->bites_on_defectors == 1
        && env->bites_at_risk == 1 && env->bites_by[0] == 1,
        "bite on a recent low-stock eater counts as eater/defector/at-risk (%d %d %d %d)",
        env->bites, env->bites_on_eaters, env->bites_on_defectors, env->bites_at_risk);
    dict_clear(&kw);
}

static void test_patchy_commons(void) {
    printf("-- commons regrowth inside the episode's patches\n");
    Dict kw = {0};
    kw_base(&kw, 1);
    kw_harvest(&kw);
    dict_set(&kw, "regrow_mode", 2);
    dict_set(&kw, "regrow_p_max", 0.2);
    dict_set(&kw, "food_distribution", FOOD_PATCHY);
    dict_set(&kw, "num_patches", 2);
    dict_set(&kw, "patch_radius", 8);
    dict_set(&kw, "patch_radius_std", 0);
    dict_set(&kw, "regrow_in_patches", 1);
    Harness h = make(&kw, 1);
    Env* env = h.env;
    CHECK(env->num_patches_ep == 2, "num_patches overrides the density count (%d patches)", env->num_patches_ep);
    place_fish(env, 0, 5.0f, 5.0f, 0.0f);
    set_action(&h, 0, -8.0f, 0.0f, 1.0f, -1.0f);
    for (int f = 0; f < 32; f++) {
        env->food[f].active = false;
    }
    env->food_active = 32;
    for (int t = 0; t < 40; t++) {
        puf_step(env);
    }
    int outside = 0;
    for (int f = 0; f < env->num_food; f++) {
        bool inside = false;
        for (int p = 0; p < env->num_patches_ep; p++) {
            float dx = env->food[f].pos.x - env->patch_center[p].x;
            float dy = env->food[f].pos.y - env->patch_center[p].y;
            inside |= dx * dx + dy * dy <= env->patch_radius_ep[p] * env->patch_radius_ep[p] + 1e-3f;
        }
        // Pellets clamped to the wall may sit just outside the disk.
        bool clamped = env->food[f].pos.x <= 0.0f || env->food[f].pos.y <= 0.0f
            || env->food[f].pos.x >= env->arena_size_x || env->food[f].pos.y >= env->arena_size_y;
        outside += env->food[f].active && !inside && !clamped;
    }
    CHECK(env->regrown > 0 && outside == 0, "all %d regrown pellets landed inside a patch (%d outside)",
        env->regrown, outside);
    dict_clear(&kw);
    // food_start in a commons mode: the episode starts with S0 active pellets.
    Dict kw2 = {0};
    kw_base(&kw2, 1);
    kw_harvest(&kw2);
    dict_set(&kw2, "regrow_mode", 2);
    dict_set(&kw2, "regrow_p_max", 0.01);
    dict_set(&kw2, "food_start", 24);
    Harness h2 = make(&kw2, 1);
    CHECK(h2.env->food_active == 24 && h2.env->num_food == 64, "commons food_start: %d of %d active at reset",
        h2.env->food_active, h2.env->num_food);
    dict_clear(&kw2);
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
    CHECK(env->agents[0].observations[OBS_SIZE - 7] == 0.0f, "obs[extra] is the constant 0 in baseline mode");
    dict_clear(&kw);
}

// ---------------------------------------------------------------------------------------------
// Allelopathic Harvest unit checks U1-U10 (docs/wef-allelopathic-harvest-v0-design.md 7.1), the
// episode Log (5.1) and the V3 trace (5.2). Everything runs on both builds with min(8, MAX_AGENTS)
// fish in the AH-strict preset (kw_allelo) with ripening switched off unless a check needs it, so
// the bushes a check places by hand stay as placed. Fish are A-tasting in slots < taste_n_a.

#define AH_FISH (MAX_AGENTS < 8 ? MAX_AGENTS : 8)
#define AH_UPPER (2.0f * 0.6745f)   // raw[3] in the upper bin: the own type at plant_bin_order 0
#define AH_LOWER (0.5f * 0.6745f)   // lower bin (still > 0, so it is a bite): the other type

static void kw_ah(Dict* kw, int num_fish, int n_a) {
    kw_base(kw, num_fish);
    kw_allelo(kw);
    dict_set(kw, "num_agents", num_fish);
    dict_set(kw, "taste_n_a", n_a);
    dict_set(kw, "ripen_lin", 0);
}

static void ah_bush(Env* env, int f, float x, float y, int type, bool ripe) {
    env->food[f] = (FishFood){.pos = {x, y}, .active = true, .type = (int8_t)type, .ripe = ripe};
    wef_bush_scale(env, &env->food[f]);
}

// n_type / n_ripe / food_active from the slots, after bushes were placed or removed by hand.
static void ah_recount(Env* env) {
    env->n_type[0] = 0;
    env->n_type[1] = 0;
    env->n_ripe = 0;
    env->food_active = 0;
    for (int f = 0; f < env->num_food; f++) {
        if (!env->food[f].active) {
            continue;
        }
        env->n_type[env->food[f].type]++;
        env->n_ripe += env->food[f].ripe;
        env->food_active++;
    }
}

static int ah_count_ripe(const Env* env) {
    int n = 0;
    for (int f = 0; f < env->num_food; f++) {
        n += env->food[f].active && env->food[f].ripe;
    }
    return n;
}

// Stationary, emitting, non-biting fish parked along the far wall (y = 36), 4 cm apart.
static void ah_park(Harness* h, int from) {
    for (int i = from; i < h->env->num_agents; i++) {
        place_fish(h->env, i, 4.0f + 4.0f * (float)i, 36.0f, 0.0f);
        set_action(h, i, -8.0f, 0.0f, 1.0f, -1.0f);
    }
}

static void ah_still(Harness* h, int i) {
    set_action(h, i, -8.0f, 0.0f, 1.0f, -1.0f);
}

// Chin sensors = mormyromasts 0..9 (within +-30 deg of the heading).
static int chin_count(const obs_t* obs, bool negative) {
    int n = 0;
    for (int s = 0; s < 10; s++) {
        n += negative ? obs[s] < 0.0f : obs[s] > 0.0f;
    }
    return n;
}

static void print_chin(const char* label, const obs_t* obs) {
    printf("     %s chin:", label);
    for (int s = 0; s < 10; s++) {
        printf(" %+.2f", obs[s]);
    }
    printf("\n");
}

// Largest distance (0.1 cm grid from 1.5) at which a bush of the given kind at `bearing` from the
// heading still produces a nonzero mormyromast reading for fish 0 (the env's class cut included).
static float ah_morm_range(Env* env, int type, bool ripe, float bearing) {
    float last = 0.0f;
    for (float d = 1.5f; d < 9.0f; d += 0.1f) {
        float x = env->fish[0].pos.x + d * cosf(env->fish[0].orientation + bearing);
        float y = env->fish[0].pos.y + d * sinf(env->fish[0].orientation + bearing);
        ah_bush(env, 0, x, y, type, ripe);
        ah_recount(env);
        compute_observations(env);
        if (morm_peak(env) == 0.0f) {
            break;
        }
        last = d;
    }
    return last;
}

// U1: observation layout.
static void test_ah_layout(void) {
    printf("-- AH U1: observation layout (MAX_AGENTS %d, OBS_SIZE %d)\n", MAX_AGENTS, OBS_SIZE);
    Dict kw = {0};
    kw_ah(&kw, AH_FISH, AH_FISH / 2);
    const int stride = OBS_SIZE + 8;
    Harness h = make_ex(&kw, AH_FISH, stride);
    Env* env = h.env;
    CHECK(OBS_SIZE == 71 + 13 * (MAX_AGENTS - 1) && OBS_META_START == 60 + 12 * (MAX_AGENTS - 1),
        "OBS_SIZE = 71 + 13 (MAX_AGENTS - 1) = %d; metadata starts at %d, last action at %d, extra at %d",
        OBS_SIZE, OBS_META_START, OBS_ACT_START, OBS_EXTRA);
    // every row writes exactly OBS_SIZE floats: sentinel fill, 8 guard words after each row
    for (int k = 0; k < AH_FISH * stride; k++) {
        h.obs[k] = -777.0f;
    }
    compute_observations(env);
    int unwritten = 0;
    int overrun = 0;
    for (int i = 0; i < AH_FISH; i++) {
        for (int s = 0; s < OBS_SIZE; s++) {
            unwritten += h.obs[i * stride + s] == -777.0f;
        }
        for (int s = OBS_SIZE; s < stride; s++) {
            overrun += h.obs[i * stride + s] != -777.0f;
        }
    }
    CHECK(unwritten == 0 && overrun == 0, "every row writes exactly %d floats (%d slots unwritten, %d guard words touched)",
        OBS_SIZE, unwritten, overrun);
    // metadata slots: -1 for a silent conspecific, a clamped size difference for a detected one
    obs_t* o0 = env->agents[0].observations;
    env->fish[AH_FISH - 1].emits_eod = false;
    compute_observations(env);
    bool meta_ok = true;
    for (int c = 0; c < MAX_AGENTS - 1; c++) {
        float v = o0[OBS_META_START + c];
        bool silent = c == AH_FISH - 2;  // fish 0 sees fish c + 1 in metadata slot c
        meta_ok = meta_ok && (silent ? v == -1.0f : (v != -1.0f && fabsf(v) <= 1.0f));
    }
    CHECK(meta_ok, "metadata slots %d-%d: -1 for the silent fish, a size difference for every detected one",
        OBS_META_START, OBS_META_START + MAX_AGENTS - 2);
    env->fish[AH_FISH - 1].emits_eod = true;
    // slot OBS_EXTRA = (n_A - n_B) / num_food under obs_extra 5
    compute_observations(env);
    CHECK(env->n_type[0] == 32 && env->n_type[1] == 32 && o0[OBS_EXTRA] == 0.0f,
        "slot %d = (n_A - n_B)/64 = 0 at the even start (obs_extra 5)", OBS_EXTRA);
    for (int f = 32; f < 40; f++) {
        env->food[f].type = 0;
    }
    ah_recount(env);
    compute_observations(env);
    CHECK(o0[OBS_EXTRA] == 0.25f, "slot %d = (40 - 24)/64 = 0.25 after 8 conversions (%.4f)", OBS_EXTRA, o0[OBS_EXTRA]);
    // slot OBS_ACT_START + 3 reports 0 / 0.5 (lower bin) / 1.0 (upper bin)
    clear_objects(env);
    ah_recount(env);
    place_fish(env, 0, 20.0f, 20.0f, 0.0f);
    ah_park(&h, 1);
    ah_still(&h, 0);
    puf_step(env);
    float s_none = o0[OBS_ACT_START + 3];
    set_action(&h, 0, -8.0f, 0.0f, 1.0f, AH_LOWER);
    puf_step(env);
    float s_lower = o0[OBS_ACT_START + 3];
    ah_still(&h, 0);
    for (int t = 0; t < 5; t++) {
        puf_step(env);  // bite cooldown
    }
    set_action(&h, 0, -8.0f, 0.0f, 1.0f, AH_UPPER);
    puf_step(env);
    float s_upper = o0[OBS_ACT_START + 3];
    CHECK(s_none == 0.0f && s_lower == 0.5f && s_upper == 1.0f,
        "slot %d reads 0 / 0.5 (lower bin) / 1.0 (upper bin): %.1f %.1f %.1f", OBS_ACT_START + 3, s_none, s_lower, s_upper);
    CHECK(o0[OBS_OWN_SIZE] == env->fish[0].size && o0[OBS_OWN_SIZE] >= 0.58f && o0[OBS_OWN_SIZE] <= 0.62f,
        "slot %d = own size %.3f (band A)", OBS_OWN_SIZE, o0[OBS_OWN_SIZE]);
    destroy(&h);
    dict_clear(&kw);
}

// U2: ripening rate and ripen_min_steps.
static void test_ah_ripening(void) {
    printf("-- AH U2: ripening rate F(x) = ripen_lin x and ripen_min_steps\n");
    for (int pass = 0; pass < 2; pass++) {
        float frac_a = pass == 0 ? 1.0f : 0.5f;
        Dict kw = {0};
        kw_ah(&kw, 1, 1);
        dict_set(&kw, "ripen_lin", 8.9e-3);
        dict_set(&kw, "start_frac_a", frac_a);
        dict_set(&kw, "episode_length", 8192);
        Harness h = make(&kw, 1);
        Env* env = h.env;
        place_fish(env, 0, 2.0f, 2.0f, 0.0f);
        ah_still(&h, 0);
        const int steps = 4000;
        long bush_steps = 0;
        long events = 0;
        bool n_ripe_ok = true;
        for (int t = 0; t < steps; t++) {
            // every bush unripe and eligible at the start of the step: 64 Bernoulli draws per step
            for (int f = 0; f < env->num_food; f++) {
                FishFood* b = &env->food[f];
                b->ripe = false;
                b->ripen_wait = 0;
                b->moment_set = false;
                wef_bush_scale(env, b);
            }
            env->n_ripe = 0;
            int before = env->ripened;
            puf_step(env);
            bush_steps += env->num_food;
            events += env->ripened - before;
            n_ripe_ok = n_ripe_ok && env->n_ripe == ah_count_ripe(env);
        }
        double rate = (double)events / (double)bush_steps;
        double f_x = 8.9e-3 * frac_a;  // x = n_type / 64 = frac_a for both types
        CHECK(fabs(rate - f_x) <= 0.10 * f_x, "x = %.1f: %ld ripenings over %ld bush-steps, rate %.5f vs F %.5f (%+.1f%%)",
            frac_a, events, bush_steps, rate, f_x, 100.0 * (rate / f_x - 1.0));
        CHECK(n_ripe_ok, "n_ripe equals the ripe count after every step");
        destroy(&h);
        dict_clear(&kw);
    }
    {
        Dict kw = {0};
        kw_ah(&kw, 1, 1);
        dict_set(&kw, "ripen_lin", 1.0);        // p = 1 at x = 1: every eligible unripe bush ripens each step
        dict_set(&kw, "start_frac_a", 1.0);
        dict_set(&kw, "ripen_min_steps", 50);
        Harness h = make(&kw, 1);
        Env* env = h.env;
        clear_objects(env);
        for (int f = 1; f < env->num_food; f++) {
            ah_bush(env, f, 30.0f + (float)(f % 8), 30.0f + (float)(f / 8), 0, false);  // far grid
        }
        ah_bush(env, 0, 11.5f, 20.0f, 0, true);  // ripe A, 1.5 cm ahead of the fish
        ah_recount(env);
        place_fish(env, 0, 10.0f, 20.0f, 0.0f);
        ah_still(&h, 0);
        puf_step(env);
        CHECK(env->food_by[0] == 1 && !env->food[0].ripe && env->food[0].ripen_wait == 49,
            "an eat sets ripen_wait = ripen_min_steps 50 (%d after this step's decrement)", env->food[0].ripen_wait);
        CHECK(env->n_ripe == 63 && env->ripened == 63, "the 63 eligible bushes ripened at p = 1 (n_ripe %d)", env->n_ripe);
        for (int t = 0; t < 49; t++) {
            puf_step(env);
        }
        CHECK(!env->food[0].ripe, "the eaten bush is still unripe 49 steps later (p = 1 blocked by ripen_min_steps)");
        puf_step(env);
        CHECK(env->food[0].ripe && env->n_ripe == 64, "and ripens on the 50th step after the eat");
        destroy(&h);
        dict_clear(&kw);
    }
}

// U3: eating.
static void test_ah_eating(void) {
    printf("-- AH U3: eating ripe bushes, taste rewards\n");
    Dict kw = {0};
    kw_ah(&kw, 2, 1);  // slot 0 tastes A, slot 1 B
    Harness h = make(&kw, 2);
    Env* env = h.env;
    clear_objects(env);
    ah_bush(env, 0, 11.5f, 20.0f, 0, true);   // ripe A 1.5 cm ahead of fish 0
    ah_bush(env, 1, 31.5f, 20.0f, 0, true);   // ripe A 1.5 cm ahead of fish 1
    ah_bush(env, 2, 20.0f, 5.0f, 1, false);   // an unripe B elsewhere
    ah_recount(env);
    place_fish(env, 0, 10.0f, 20.0f, 0.0f);
    place_fish(env, 1, 30.0f, 20.0f, 0.0f);
    ah_still(&h, 0);
    ah_still(&h, 1);
    CHECK(env->fish[0].taste == 0 && env->fish[1].taste == 1, "taste_n_a 1: slot 0 tastes A, slot 1 B");
    CHECK(env->n_ripe == 2 && env->n_type[0] == 2 && env->n_type[1] == 1, "n_ripe / n_type count the placed bushes");
    puf_step(env);
    CHECK(h.rew[0] == 1.0f, "A fish eating a ripe A bush: reward %.3f (taste_match 1.0)", h.rew[0]);
    CHECK(h.rew[1] == 0.5f, "B fish eating a ripe A bush: reward %.3f (taste_other 0.5)", h.rew[1]);
    CHECK(env->food[0].active && !env->food[0].ripe && env->food[0].type == 0 && env->food[1].active && !env->food[1].ripe,
        "eaten bushes stay active, turn unripe in place and keep their type");
    CHECK(env->n_ripe == 0 && env->food_eaten == 2 && env->food_by[0] == 1 && env->food_by[1] == 1 && env->eats_match == 1,
        "n_ripe 0 after both eats, food_eaten 2, eats_match 1");
    CHECK(env->fish[0].trace_ate_type == 1 && env->fish[0].last_eaten_slot == 0 && env->fish[0].last_eat_tick == env->tick,
        "ate_type / last_eaten_slot / last_eat_tick recorded (trace, plant_proactive)");
    CHECK(env->fish[0].eat_cooldown == 2 && env->fish[0].pos.x == 10.0f, "eat cooldown 3 (2 after decrement) pins the fish");
    for (int t = 0; t < 3; t++) {
        puf_step(env);
    }
    CHECK(h.rew[0] == 0.0f && env->food_by[0] == 1 && !env->food[0].ripe,
        "an unripe bush 1.5 cm ahead is not eaten once the cooldown expired (reward %.3f)", h.rew[0]);
    env->taste_other_b = 0.25f;  // per-group intensity (the paper's Fig 6A knob)
    env->food[1].ripe = true;
    env->food[1].moment_set = false;
    wef_bush_scale(env, &env->food[1]);
    ah_recount(env);
    puf_step(env);
    CHECK(h.rew[1] == 0.25f && env->food_by[1] == 2, "taste_other_b 0.25 overrides taste_other for B fish: reward %.3f", h.rew[1]);
    destroy(&h);
    dict_clear(&kw);
}

// U4 planting (with U9 normalisers and U10 zap cooldown where they arise).
static void test_ah_planting(void) {
    printf("-- AH U4: planting, the hold, zaps on a held planter (U9 normalisers, U10 zap cooldown)\n");
    Dict kw = {0};
    kw_ah(&kw, 2, 1);                          // slot 0 A, slot 1 B
    dict_set(&kw, "bitten_freeze_steps", 25);  // zap arm
    Harness h = make(&kw, 2);
    Env* env = h.env;
    obs_t* o0 = env->agents[0].observations;
    obs_t* o1 = env->agents[1].observations;

    // (a) conversion and the hold
    clear_objects(env);
    ah_bush(env, 0, 12.0f, 20.0f, 1, false);  // unripe B, 2 cm ahead
    ah_bush(env, 1, 11.5f, 20.0f, 0, true);   // ripe A 1.5 cm ahead: edible, but not during the hold
    ah_recount(env);
    place_fish(env, 0, 10.0f, 20.0f, 0.0f);
    place_fish(env, 1, 30.0f, 36.0f, 0.0f);
    set_action(&h, 0, -8.0f, 0.0f, 1.0f, AH_UPPER);
    ah_still(&h, 1);
    puf_step(env);
    CHECK(env->food[0].type == 0 && !env->food[0].ripe && env->n_type[0] == 2 && env->n_type[1] == 0,
        "own-bin bite 2 cm from an unripe B bush converts it to A (n_type %d/%d)", env->n_type[0], env->n_type[1]);
    CHECK(env->plantings == 1 && env->plantings_by[0] == 1 && env->plantings_g[0] == 1 && env->plant_own == 1
        && env->plant_proactive == 1 && env->plant_attempts == 1 && env->plant_noop == 0 && env->food_by[0] == 0,
        "plantings 1 (slot 0, taste A, own type, proactive), plant_attempts 1, the bite skipped eating");
    CHECK(env->fish[0].eat_cooldown == 31 && env->fish[0].bite_cooldown == 31,
        "hold: eat_cooldown = plant_steps 32 (31 after decrement), bite_cooldown = max(5, 32) (%d / %d)",
        env->fish[0].eat_cooldown, env->fish[0].bite_cooldown);
    CHECK(env->fish[0].trace_plant_type == 1 && env->fish[0].trace_planted_slot == 0 && env->fish[0].last_plant_type == 0
        && env->fish[0].plant_mark == env->tick, "plant_type 1 (A) / planted_slot 0 / plant_mark recorded");
    CHECK(o0[OBS_EAT_CD] == 31.0f / 32.0f && o0[OBS_BITE_CD] == 31.0f / 32.0f,
        "U9: slot %d = 31/32 (hold) and slot %d = 31/32 (bite cooldown over cd_max 32): both <= 1", OBS_EAT_CD, OBS_BITE_CD);
    set_action(&h, 0, 8.0f, 0.0f, 1.0f, -1.0f);  // tries to swim off at full speed
    float x0 = env->fish[0].pos.x;
    int moved_at = -1;
    int ate_at = -1;
    for (int t = 1; t <= 40 && ate_at < 0; t++) {
        puf_step(env);
        if (moved_at < 0 && env->fish[0].pos.x != x0) {
            moved_at = t;
        }
        if (env->food_by[0] == 1) {
            ate_at = t;
        }
    }
    CHECK(ate_at == 32 && moved_at < 0,
        "the hold blocks eating and motion: the ripe bush 1.5 cm ahead is eaten on step %d after the planting, no motion before (moved_at %d)",
        ate_at, moved_at);
    clear_objects(env);
    ah_recount(env);
    puf_step(env);
    puf_step(env);
    CHECK(env->fish[0].pos.x == x0, "still pinned by the 3-step eat cooldown");
    puf_step(env);
    CHECK(env->fish[0].pos.x > x0 + 0.5f, "motion resumes once eat_cooldown expires (x %.2f -> %.2f)", x0, env->fish[0].pos.x);

    // (b) a zap during the hold: the planter is not immune, the freeze stacks on the remaining hold
    clear_objects(env);
    ah_bush(env, 0, 12.0f, 20.0f, 1, false);
    ah_recount(env);
    place_fish(env, 0, 10.0f, 20.0f, 0.0f);
    place_fish(env, 1, 7.5f, 20.0f, 0.0f);   // B fish 2.5 cm behind the planter, facing it (in the 3 cm cone, no body contact)
    set_action(&h, 0, -8.0f, 0.0f, 1.0f, AH_UPPER);
    ah_still(&h, 1);
    puf_step(env);
    CHECK(env->plantings == 2 && env->fish[0].eat_cooldown == 31, "second conversion, hold running (%d left)", env->fish[0].eat_cooldown);
    ah_still(&h, 0);
    set_action(&h, 1, -8.0f, 0.0f, 1.0f, AH_UPPER);  // fish first (plant_priority 0): a zap
    int bites_before = env->bites;
    puf_step(env);
    CHECK(env->fish[1].bite_victim == 0 && env->fish[0].was_bitten && env->bites == bites_before + 1
        && env->zaps_cross == 1 && env->zaps_same == 0 && env->freezes == 1,
        "a held planter is zappable; the B-on-A zap counts as zaps_cross");
    CHECK(env->fish[0].freeze == 55, "freeze = 25 + remaining hold 31 = 56 (55 after decrement): got %d", env->fish[0].freeze);
    CHECK(env->fish[1].bite_cooldown == 24, "U10: attacker bite_cooldown = zap_cooldown_steps 25 (24 after decrement): got %d",
        env->fish[1].bite_cooldown);
    CHECK(o1[OBS_BITE_CD] == 24.0f / 32.0f && o0[OBS_EAT_CD] == 1.0f && o0[OBS_BITTEN] == 1.0f,
        "U9: attacker slot %d = 24/32 (<= 1); victim slot %d = 1 while frozen, slot %d = 1 (zapped this step)",
        OBS_BITE_CD, OBS_EAT_CD, OBS_BITTEN);
    CHECK(env->plant_attempts == 2 && h.rew[0] == 0.0f && h.rew[1] == 0.0f,
        "a zap is not a plant attempt; bitten_reward / bite_reward 0 in the preset");
    float xz = env->fish[0].pos.x;
    int free_at = -1;
    set_action(&h, 0, 8.0f, 0.0f, 1.0f, -1.0f);  // tries to swim off at full speed
    ah_still(&h, 1);
    for (int t = 1; t <= 70 && free_at < 0; t++) {
        puf_step(env);
        if (env->fish[0].pos.x != xz) {
            free_at = t;
        }
    }
    CHECK(free_at == 56, "the frozen planter moves again %d steps after the zap (25 + 31)", free_at);

    // (c) a planting bite on a ripe bush: nothing but the 5-step cooldown and a plant_attempt
    clear_objects(env);
    ah_bush(env, 0, 12.0f, 20.0f, 1, true);
    ah_recount(env);
    place_fish(env, 0, 10.0f, 20.0f, 0.0f);
    place_fish(env, 1, 30.0f, 36.0f, 0.0f);
    set_action(&h, 0, -8.0f, 0.0f, 1.0f, AH_UPPER);
    ah_still(&h, 1);
    int pa = env->plant_attempts;
    int pn = env->plant_noop;
    int pl = env->plantings;
    int fb = env->food_by[0];
    puf_step(env);
    CHECK(env->food[0].type == 1 && env->food[0].ripe && env->plantings == pl && env->plant_noop == pn
        && env->plant_attempts == pa + 1 && env->fish[0].bite_cooldown == 4 && env->fish[0].eat_cooldown == 0
        && env->food_by[0] == fb,
        "bite on a ripe bush: no conversion, no hold, 5-step cooldown (4 after decrement), plant_attempts +1, not eaten");

    // (d) same-type-only cone: plant_noop, no hold
    ah_bush(env, 0, 12.0f, 20.0f, 0, false);
    ah_recount(env);
    place_fish(env, 0, 10.0f, 20.0f, 0.0f);
    puf_step(env);
    CHECK(env->plant_noop == pn + 1 && env->plantings == pl && env->plant_attempts == pa + 2
        && env->fish[0].eat_cooldown == 0 && env->fish[0].bite_cooldown == 4,
        "same-type-only cone: plant_noop +1, no hold, plant_cooldown 5 (4 after decrement)");

    // (e) target rule
    clear_objects(env);
    ah_bush(env, 0, 11.5f, 20.0f, 0, false);  // same-type unripe at 1.5 cm
    ah_bush(env, 1, 12.5f, 20.0f, 1, false);  // off-type unripe at 2.5 cm
    ah_recount(env);
    place_fish(env, 0, 10.0f, 20.0f, 0.0f);
    puf_step(env);
    CHECK(env->food[1].type == 0 && env->food[0].type == 0 && env->plantings == pl + 1 && env->plant_noop == pn + 1
        && env->fish[0].trace_planted_slot == 1,
        "target rule: the off-type bush at 2.5 cm converts although a same-type one sits at 1.5 cm; plant_noop unchanged");
    clear_objects(env);
    ah_bush(env, 0, 11.5f, 20.0f, 1, true);   // ripe off-type at 1.5 cm
    ah_bush(env, 1, 12.5f, 20.0f, 1, false);  // unripe off-type at 2.5 cm
    ah_recount(env);
    place_fish(env, 0, 10.0f, 20.0f, 0.0f);
    puf_step(env);
    CHECK(env->food[0].type == 1 && env->food[0].ripe && env->food[1].type == 0 && env->plantings == pl + 2,
        "ripe bushes do not block: the unripe B at 2.5 cm converts, the ripe B at 1.5 cm is untouched");

    // (f) plantings_a / plantings_b follow the planter's taste
    clear_objects(env);
    ah_bush(env, 0, 32.0f, 20.0f, 0, false);  // unripe A 2 cm ahead of the B fish
    ah_recount(env);
    place_fish(env, 0, 10.0f, 20.0f, 0.0f);
    place_fish(env, 1, 30.0f, 20.0f, 0.0f);
    ah_still(&h, 0);
    set_action(&h, 1, -8.0f, 0.0f, 1.0f, AH_UPPER);
    int ga = env->plantings_g[0];
    int gb = env->plantings_g[1];
    puf_step(env);
    CHECK(env->food[0].type == 1 && env->plantings_g[1] == gb + 1 && env->plantings_g[0] == ga && env->plant_own_g[1] == 1
        && env->fish[1].trace_plant_type == 2,
        "a B planter's conversion counts in plantings_b (A planters' %d unchanged), own type, plant_type 2", ga);
    ah_still(&h, 1);

    // (g) zero-travel re-type: eat at t, bite at t + 1 converts the just-eaten bush, not proactive
    clear_objects(env);
    ah_bush(env, 0, 11.5f, 20.0f, 1, true);   // ripe B 1.5 cm ahead of the A fish
    ah_recount(env);
    place_fish(env, 0, 10.0f, 20.0f, 0.0f);
    place_fish(env, 1, 30.0f, 36.0f, 0.0f);
    ah_still(&h, 0);
    puf_step(env);
    CHECK(h.rew[0] == 0.5f && !env->food[0].ripe && env->food[0].type == 1 && env->fish[0].pos.x == 10.0f,
        "eat at t: reward 0.5, bush unripe B, fish pinned by the eat cooldown");
    int pro = env->plant_proactive;
    set_action(&h, 0, 8.0f, 0.0f, 1.0f, AH_UPPER);
    puf_step(env);
    CHECK(env->food[0].type == 0 && env->fish[0].pos.x == 10.0f && env->plant_proactive == pro && env->fish[0].eat_cooldown == 31
        && env->plantings == pl + 4,
        "bite at t + 1 converts the just-eaten bush with zero travel; plant_proactive unchanged; the hold replaces the eat cooldown");

    // (h) plant_bin_order 1 swaps the bins; plant_mode 2 ignores them; plant_mode 0 disables planting
    env->plant_bin_order = 1;
    clear_objects(env);
    ah_bush(env, 0, 12.0f, 20.0f, 1, false);
    ah_recount(env);
    place_fish(env, 0, 10.0f, 20.0f, 0.0f);
    set_action(&h, 0, -8.0f, 0.0f, 1.0f, AH_UPPER);
    pn = env->plant_noop;
    pl = env->plantings;
    puf_step(env);
    CHECK(env->food[0].type == 1 && env->plant_noop == pn + 1 && env->plantings == pl,
        "plant_bin_order 1: the upper bin now plants the other type (B on an unripe B: plant_noop)");
    place_fish(env, 0, 10.0f, 20.0f, 0.0f);
    set_action(&h, 0, -8.0f, 0.0f, 1.0f, AH_LOWER);
    puf_step(env);
    CHECK(env->food[0].type == 0 && env->plantings == pl + 1, "plant_bin_order 1: the lower bin plants the own type");
    env->plant_bin_order = 0;
    env->plant_mode = 2;
    clear_objects(env);
    ah_bush(env, 0, 12.0f, 20.0f, 1, false);
    ah_recount(env);
    place_fish(env, 0, 10.0f, 20.0f, 0.0f);
    set_action(&h, 0, -8.0f, 0.0f, 1.0f, AH_LOWER);
    puf_step(env);
    CHECK(env->food[0].type == 0 && env->plantings == pl + 2, "plant_mode 2 ignores the bin: a lower-bin bite plants the own type");
    env->plant_mode = 0;
    clear_objects(env);
    ah_bush(env, 0, 12.0f, 20.0f, 1, false);
    ah_recount(env);
    place_fish(env, 0, 10.0f, 20.0f, 0.0f);
    set_action(&h, 0, -8.0f, 0.0f, 1.0f, AH_UPPER);
    pa = env->plant_attempts;
    puf_step(env);
    CHECK(env->food[0].type == 1 && env->plantings == pl + 2 && env->plant_noop == pn + 1 && env->plant_attempts == pa + 1
        && env->fish[0].bite_cooldown == 4 && env->fish[0].eat_cooldown == 0,
        "plant_mode 0: a bite on a bush is a no-op that spends the 5-step cooldown (still a plant_attempt)");
    env->plant_mode = 1;
    destroy(&h);
    dict_clear(&kw);

    // (i) bitten_freeze_steps 0 (learnability rungs): a zap on a held planter does not freeze it at all
    Dict kw2 = {0};
    kw_ah(&kw2, 2, 1);
    Harness h2 = make(&kw2, 2);
    Env* e2 = h2.env;
    clear_objects(e2);
    ah_bush(e2, 0, 12.0f, 20.0f, 1, false);
    ah_recount(e2);
    place_fish(e2, 0, 10.0f, 20.0f, 0.0f);
    place_fish(e2, 1, 7.5f, 20.0f, 0.0f);
    set_action(&h2, 0, -8.0f, 0.0f, 1.0f, AH_UPPER);
    ah_still(&h2, 1);
    puf_step(e2);
    ah_still(&h2, 0);
    set_action(&h2, 1, -8.0f, 0.0f, 1.0f, AH_UPPER);
    puf_step(e2);
    CHECK(e2->bites == 1 && e2->fish[0].was_bitten && e2->fish[0].freeze == 0 && e2->freezes == 0
        && e2->fish[0].eat_cooldown == 30 && e2->zaps_cross == 1,
        "bitten_freeze_steps 0: the zap lands but freezes nothing (hold continues, %d left)", e2->fish[0].eat_cooldown);
    CHECK(h2.rew[0] == 0.0f && e2->fish[1].bite_cooldown == 24,
        "bitten_reward 0; zap cooldown 25 (24 after decrement) regardless of the freeze setting");
    destroy(&h2);
    dict_clear(&kw2);

    // (j) the freeze stacks on a planting HOLD only (Q14): a victim serving the plain 3-step eat
    // cooldown is frozen for bitten_freeze_steps exactly; (k) U9 with zap_steps > plant_steps:
    // the attacker's recovery hold reads <= 1 in slot OBS_SIZE - 1 (normaliser covers zap_steps).
    Dict kw3 = {0};
    kw_ah(&kw3, 2, 1);
    dict_set(&kw3, "bitten_freeze_steps", 25);
    dict_set(&kw3, "zap_steps", 40);           // > plant_steps 32
    Harness h3 = make(&kw3, 2);
    Env* e3 = h3.env;
    obs_t* p0 = e3->agents[0].observations;
    obs_t* p1 = e3->agents[1].observations;
    clear_objects(e3);
    ah_bush(e3, 0, 11.5f, 20.0f, 1, true);    // ripe B 1.5 cm ahead of the A fish: eaten at t
    ah_recount(e3);
    place_fish(e3, 0, 10.0f, 20.0f, 0.0f);
    place_fish(e3, 1, 7.5f, 20.0f, 0.0f);     // B fish 2.5 cm behind, facing it
    ah_still(&h3, 0);
    ah_still(&h3, 1);
    puf_step(e3);
    CHECK(e3->food_by[0] == 1 && e3->fish[0].eat_cooldown == 2 && e3->fish[0].trace_ate_type == 2,
        "eat at t: the ordinary eat cooldown 3 (2 after decrement) is running, trace hold %d (not a hold)",
        e3->fish[0].eat_cooldown > EAT_COOLDOWN_STEPS ? e3->fish[0].eat_cooldown : 0);
    set_action(&h3, 1, -8.0f, 0.0f, 1.0f, AH_UPPER);
    puf_step(e3);
    CHECK(e3->fish[1].bite_victim == 0 && e3->fish[0].was_bitten && e3->freezes == 1,
        "zap at t + 1 on a fish serving the plain eat cooldown");
    CHECK(e3->fish[0].freeze == 24, "freeze = bitten_freeze_steps 25 exactly (24 after decrement), no stacking on a plain eat cooldown: got %d",
        e3->fish[0].freeze);
    CHECK(e3->fish[1].eat_cooldown == 39 && p1[OBS_EAT_CD] == 39.0f / 40.0f && p1[OBS_EAT_CD] <= 1.0f,
        "U9: zap_steps 40 > plant_steps 32: attacker eat_cooldown 39, slot %d = 39/40 (<= 1): got %d, %.3f",
        OBS_EAT_CD, e3->fish[1].eat_cooldown, p1[OBS_EAT_CD]);
    CHECK(p0[OBS_EAT_CD] == 1.0f && p1[OBS_BITE_CD] == 24.0f / 32.0f,
        "victim slot %d = 1 while frozen; attacker bite cooldown 24/32", OBS_EAT_CD);
    // the attacker's zap recovery is a hold (eat_cooldown > 3): a zap on IT would stack
    ah_still(&h3, 1);
    place_fish(e3, 0, 30.0f, 36.0f, 0.0f);
    e3->fish[0].freeze = 0;
    place_fish(e3, 1, 20.0f, 20.0f, 0.0f);    // (place_fish clears the cooldowns and the freeze)
    place_fish(e3, 0, 17.5f, 20.0f, 0.0f);    // the thawed A fish behind the B fish
    e3->fish[1].eat_cooldown = 20;            // mid-recovery (zap_steps 40) or mid-hold: > 3
    set_action(&h3, 0, -8.0f, 0.0f, 1.0f, AH_UPPER);
    int cd1 = e3->fish[1].eat_cooldown;
    puf_step(e3);
    CHECK(e3->fish[0].bite_victim == 1 && cd1 == 20 && e3->fish[1].freeze == 25 + cd1 - 1,
        "a zap on a fish with eat_cooldown %d (> 3: a hold) stacks: freeze %d = 25 + %d - 1",
        cd1, e3->fish[1].freeze, cd1);
    destroy(&h3);
    dict_clear(&kw3);
}

// Episode Log (section 5.1): one conversion at tick 1, one zap at tick 2, 64-step episode.
static void test_ah_log(void) {
    printf("-- AH Log: episode metrics\n");
    Dict kw = {0};
    kw_ah(&kw, 2, 1);
    dict_set(&kw, "episode_length", 64);
    dict_set(&kw, "bitten_freeze_steps", 25);
    Harness h = make(&kw, 2);
    Env* env = h.env;
    // keep the reset's 32 A / 32 B bushes, moved onto a far grid; bush 32 (type B) 2 cm ahead of fish 0
    for (int f = 0; f < env->num_food; f++) {
        env->food[f].pos = (Vec2){22.0f + 2.0f * (float)(f % 9), 20.0f + 2.0f * (float)(f / 9)};
    }
    env->food[32].pos = (Vec2){12.0f, 20.0f};
    ah_recount(env);
    place_fish(env, 0, 10.0f, 20.0f, 0.0f);
    place_fish(env, 1, 7.5f, 20.0f, 0.0f);   // 2.5 cm behind the planter: in the bite cone, no body contact
    set_action(&h, 0, -8.0f, 0.0f, 1.0f, AH_UPPER);
    ah_still(&h, 1);
    puf_step(env);  // tick 1: B -> A, 33 / 31
    ah_still(&h, 0);
    set_action(&h, 1, -8.0f, 0.0f, 1.0f, AH_UPPER);
    puf_step(env);  // tick 2: fish 1 zaps the held planter
    ah_still(&h, 1);
    CHECK(env->n_type[0] == 33 && env->bites == 1 && env->fish[0].freeze == 55, "scene: 33 A / 31 B, one zap, planter frozen (%d)",
        env->fish[0].freeze);
    while (env->episode == 0) {
        puf_step(env);
    }
    Log* l = &env->log;
    float m = 33.0f / 64.0f;
    CHECK(l->n == 1.0f && l->plantings == 1.0f && l->plantings_a == 1.0f && l->plantings_b == 0.0f && l->plant_attempts == 1.0f
        && l->plant_noop == 0.0f && l->plant_proactive == 1.0f,
        "plantings 1 / plantings_a 1 / plantings_b 0 / plant_attempts 1 / plant_noop 0 / plant_proactive 1");
    CHECK(l->plant_own_frac == 1.0f && l->plant_own_frac_a == 1.0f && l->plant_own_frac_b == 0.0f && l->plant_major_frac == 0.0f
        && l->plant_gini == 0.5f,
        "plant_own_frac 1 (a: 1, b: 0 with no B plantings), plant_major_frac 0 (tie before the conversion), plant_gini 0.5");
    CHECK(fabsf(l->mono_final - m) < 1e-6f && fabsf(l->frac_a_final - m) < 1e-6f && fabsf(l->mono_frac - m) < 1e-5f
        && fabsf(l->frac_a_mean - m) < 1e-5f && l->perf == l->mono_frac,
        "mono_final / frac_a_final / mono_frac / frac_a_mean = 33/64 (%.4f %.4f %.4f %.4f), perf = mono_frac",
        l->mono_final, l->frac_a_final, l->mono_frac, l->frac_a_mean);
    CHECK(fabsf(l->conv_c - 1.0f / 64.0f) < 1e-5f && l->majority_frac_final == 0.5f && l->time_to_convention == 1.0f,
        "conv_c = |33/64 - 1/2| (%.4f), majority_frac_final 0.5 (1 + 1 tie), time_to_convention 1 (never)", l->conv_c);
    CHECK(l->zaps_cross == 1.0f && l->zaps_same == 0.0f && l->bites == 1.0f && l->freezes == 1.0f
        && fabsf(l->frozen_frac - 56.0f / 128.0f) < 1e-6f,
        "zaps_cross 1, zaps_same 0, bites 1, freezes 1, frozen_frac 56/128 (%.4f)", l->frozen_frac);
    CHECK(l->taste_a_n == 1.0f && l->taste_a_return == 0.0f && l->taste_b_return == 0.0f && l->eaten_match_frac == 0.0f
        && l->ripened == 0.0f && l->ripe_frac == 0.0f && l->equality == 1.0f && l->collective_food == 0.0f && l->eod_rate == 1.0f,
        "taste_a_n 1, returns 0 (no eats, no zap rewards), ripened 0, equality 1, eod_rate 1");
    destroy(&h);
    dict_clear(&kw);
}

// U5: mormyromast type cue.
static void test_ah_sensing_type(void) {
    printf("-- AH U5: mormyromast type cue (sign), two-bush and silent-fish cases\n");
    Dict kw = {0};
    kw_ah(&kw, 2, 2);
    Harness h = make(&kw, 2);
    Env* env = h.env;
    obs_t* o0 = env->agents[0].observations;
    clear_objects(env);
    place_fish(env, 0, 20.0f, 20.0f, 0.0f);   // reader at the arena centre heading +x
    place_fish(env, 1, 36.0f, 36.0f, 0.0f);   // conspecific parked out of mormyromast range
    // (a) sign flips with the type at 3 cm (ripe bushes, r 0.30)
    ah_bush(env, 0, 23.0f, 20.0f, 0, true);
    ah_recount(env);
    compute_observations(env);
    float pa = morm_peak(env);
    env->food[0].type = 1;
    wef_bush_scale(env, &env->food[0]);
    ah_recount(env);
    compute_observations(env);
    float pb = morm_peak(env);
    CHECK(pa < 0.0f && pb > 0.0f && fabsf(pa + pb) < 0.02f, "A bush at 3 cm reads %+.3f, B bush %+.3f: sign = type, mirror images", pa, pb);
    CHECK(fabsf(pa + 0.45f) < 0.05f, "A ripe at 3 cm: chin peak %+.3f vs [field] -0.45", pa);
    // (b) two bushes of opposite type: A ripe ahead, B ripe 3 cm at +40 deg. Section 3.2 (i)
    // documents the 3 cm / 3 cm case (chin -0.40..-0.34, +0.21, +0.35): a sign change across the
    // chin array. 7.1 U5 names A at 2 cm instead; that case is measured and printed too (the
    // nearer bush's field is ~40x stronger and dominates every chin sensor, so no sign change).
    float bx = 20.0f + 3.0f * cosf(40.0f * PI_F / 180.0f);
    float by = 20.0f + 3.0f * sinf(40.0f * PI_F / 180.0f);
    int neg3 = 0;
    int pos3 = 0;
    for (int d = 3; d >= 2; d--) {
        clear_objects(env);
        ah_bush(env, 0, 20.0f + (float)d, 20.0f, 0, true);
        ah_bush(env, 1, bx, by, 1, true);
        ah_recount(env);
        compute_observations(env);
        int neg = chin_count(o0, true);
        int pos = chin_count(o0, false);
        print_chin(d == 3 ? "A 3 cm ahead + B 3 cm at +40 deg" : "A 2 cm ahead + B 3 cm at +40 deg", o0);
        if (d == 3) {
            neg3 = neg;
            pos3 = pos;
        } else {
            printf("     (7.1 U5 geometry, measured: %d chin negative, %d positive; the 2 cm bush dominates)\n", neg, pos);
        }
    }
    clear_objects(env);
    ah_bush(env, 1, bx, by, 1, true);
    ah_recount(env);
    compute_observations(env);
    print_chin("B alone (3 cm at +40 deg)       ", o0);
    int pos_b = chin_count(o0, false);
    CHECK(neg3 > 0 && pos3 > 0 && pos_b == 10,
        "two-bush case (3 cm / 3 cm, 3.2 (i)): sign change across the chin array (%d negative, %d positive); B alone all positive",
        neg3, pos3);
    // (c) silent conspecific ahead of a ripe B bush: the known failure case at 4 cm, intact at 6 cm
    clear_objects(env);
    ah_bush(env, 0, 23.0f, 20.0f, 1, true);
    ah_recount(env);
    place_fish(env, 1, 24.0f, 20.0f, 0.0f);
    env->fish[1].emits_eod = false;
    compute_observations(env);
    float p4 = morm_peak(env);
    int neg4 = chin_count(o0, true);
    print_chin("B 3 cm + silent fish 4 cm       ", o0);
    env->fish[1].pos.x = 25.0f;
    compute_observations(env);
    int neg5 = chin_count(o0, true);
    env->fish[1].pos.x = 26.0f;
    compute_observations(env);
    float p6 = morm_peak(env);
    int neg6 = chin_count(o0, true);
    print_chin("B 3 cm + silent fish 6 cm       ", o0);
    CHECK(p4 < 0.0f, "known failure case: ripe B at 3 cm + silent fish at 4 cm reads %+.3f (sign flipped, %d of 10 chin negative)", p4, neg4);
    CHECK(p6 > 0.0f && neg6 == 0, "silent fish at 6 cm: B sign intact (%+.3f, %d chin negative; at 5 cm %d negative)", p6, neg6, neg5);
    destroy(&h);
    dict_clear(&kw);
}

// U6: ampullary ripeness cue with the reader 3 cm from a wall and a conspecific 6 cm ahead.
static void test_ah_sensing_ripe(void) {
    printf("-- AH U6: ampullary ripeness cue (off-centre reader, wall at 3 cm, conspecific at 6 cm)\n");
    Dict kw = {0};
    kw_ah(&kw, 2, 2);
    Harness h = make(&kw, 2);
    Env* env = h.env;
    obs_t* o0 = env->agents[0].observations;
    clear_objects(env);
    for (int f = 1; f < env->num_food; f++) {  // 63 far A bushes so x = n_A / 64 = 1 for the ripening step below
        ah_bush(env, f, 22.0f + 2.0f * (float)((f - 1) % 9), 20.0f + 2.0f * (float)((f - 1) / 9), 0, false);
    }
    ah_bush(env, 0, 3.0f, 23.0f, 0, false);   // unripe A bush 3 cm ahead (intrinsic moment off)
    ah_recount(env);
    place_fish(env, 0, 3.0f, 20.0f, PI_F / 2.0f);   // 3 cm from the left wall, heading +y
    place_fish(env, 1, 3.0f, 26.0f, PI_F / 2.0f);   // emitting conspecific 6 cm ahead
    ah_still(&h, 0);
    ah_still(&h, 1);
    unsigned int saved = env->rng;
    compute_observations(env);
    float with_unripe[NUM_AMPULLARY];
    memcpy(with_unripe, o0 + NUM_MORMYROMASTS, sizeof(with_unripe));
    env->food[0].active = false;
    ah_recount(env);
    env->rng = saved;
    compute_observations(env);
    float without[NUM_AMPULLARY];
    memcpy(without, o0 + NUM_MORMYROMASTS, sizeof(without));
    int nonzero = 0;
    for (int s = 0; s < NUM_AMPULLARY; s++) {
        nonzero += without[s] != 0.0f;
    }
    CHECK(memcmp(with_unripe, without, sizeof(without)) == 0,
        "unripe bush (unripe_intrinsic 0): the 24 ampullary readings are identical with and without it (same RNG state)");
    CHECK(nonzero > 0, "the off-centre sensor itself reads nonzero (%d of 24: own wall image, conspecific)", nonzero);
    // ripen the bush through the env's own ripening loop (p = 1 at x = 1), nothing else moving
    env->food[0].active = true;
    ah_recount(env);
    env->ripen_lin = 1.0f;
    env->fish[0].freeze = 1;
    env->fish[1].freeze = 1;
    puf_step(env);
    CHECK(env->food[0].ripe && env->n_ripe == 64 && env->fish[0].pos.y == 20.0f && env->fish[1].pos.y == 26.0f,
        "ripening step: the bush ripened in place (n_ripe %d), fish unmoved", env->n_ripe);
    env->ripen_lin = 0.0f;
    env->rng = saved;
    compute_observations(env);
    float with_ripe[NUM_AMPULLARY];
    memcpy(with_ripe, o0 + NUM_MORMYROMASTS, sizeof(with_ripe));
    float max_diff = 0.0f;
    float peak_ripe = 0.0f;
    float peak_without = 0.0f;
    for (int s = 0; s < NUM_AMPULLARY; s++) {
        max_diff = fmaxf(max_diff, fabsf(with_ripe[s] - without[s]));
        if (fabsf(with_ripe[s]) > fabsf(peak_ripe)) {
            peak_ripe = with_ripe[s];
        }
        if (fabsf(without[s]) > fabsf(peak_without)) {
            peak_without = without[s];
        }
    }
    // the bush's own contribution at the best sensor (the env's field code on the bush's dipole alone)
    Dipole dip = {to_m(env->food[0].pos), env->food[0].intrinsic_moment};
    float contrib = 0.0f;
    for (int s = 0; s < NUM_AMPULLARY; s++) {
        Sensor w = sensor_world(&g_amp[s], &env->fish[0]);
        Vec2 f = measure_field(env, w.p, NULL, 0, &dip, 1, 0, 1, AMP_AGENT_RANGE_CM, AMP_FOOD_RANGE_CM, 0.0f,
            env->reflection_wall_range_cm);
        contrib = fmaxf(contrib, fabsf(f.x * w.n.x + f.y * w.n.y));
    }
    CHECK(contrib > AMPULLARY_MIN_VM, "ripe bush (ripe_intrinsic 3) at 3 cm contributes %.2e V/m at the best sensor (> %.0e)",
        contrib, AMPULLARY_MIN_VM);
    CHECK(memcmp(with_ripe, without, sizeof(without)) != 0 && max_diff > 0.1f,
        "ripening in place changes the reading (cache invalidated): max encoded change %.2f, peak %+.2f with vs %+.2f without",
        max_diff, peak_ripe, peak_without);
    destroy(&h);
    dict_clear(&kw);
}

// U7: the distance confound of the radius cue, preset radii and the Q15 alternative; U5's ranges.
static void test_ah_confound(void) {
    printf("-- AH U7: mormyromast ripeness cue vs distance, sensing ranges\n");
    for (int alt = 0; alt < 2; alt++) {
        Dict kw = {0};
        kw_ah(&kw, 1, 1);
        if (alt) {
            dict_set(&kw, "food_radius_unripe_cm", 0.25);
            dict_set(&kw, "food_radius_ripe_cm", 0.40);
            dict_set(&kw, "unripe_intrinsic", 1.0);
        }
        const char* label = alt ? "Q15 alternative (0.25 / 0.40, intrinsic on)" : "preset (0.12 / 0.30, unripe intrinsic off)";
        Harness h = make(&kw, 1);
        Env* env = h.env;
        obs_t* o0 = env->agents[0].observations;
        clear_objects(env);
        place_fish(env, 0, 20.0f, 20.0f, 0.0f);
        float u[NUM_MORMYROMASTS];
        float r[NUM_MORMYROMASTS];
        ah_bush(env, 0, 22.0f, 20.0f, 0, false);
        ah_recount(env);
        compute_observations(env);
        memcpy(u, o0, sizeof(u));
        float pu = morm_peak(env);
        ah_bush(env, 0, 22.7f, 20.0f, 0, true);
        ah_recount(env);
        compute_observations(env);
        memcpy(r, o0, sizeof(r));
        float pr = morm_peak(env);
        double l2 = 0.0;
        for (int s = 0; s < NUM_MORMYROMASTS; s++) {
            l2 += (double)(u[s] - r[s]) * (double)(u[s] - r[s]);
        }
        l2 = sqrt(l2);
        printf("   %s\n", label);
        print_chin("unripe at 2.0 cm", u);
        print_chin("ripe at 2.7 cm  ", r);
        CHECK(pu < 0.0f && pr < 0.0f && fabsf(fabsf(pr) - fabsf(pu)) < 0.2f,
            "distance confound: ripe at 2.7 cm (%+.3f) reads within 0.2 of unripe at 2.0 cm (%+.3f); L2 over 36 floats %.3f",
            pr, pu, l2);
        float ra_u = ah_morm_range(env, 0, false, 0.0f);
        float rb_u = ah_morm_range(env, 0, false, PI_F / 2.0f);
        float ra_r = ah_morm_range(env, 0, true, 0.0f);
        float rb_r = ah_morm_range(env, 0, true, PI_F / 2.0f);
        CHECK(ra_r >= ra_u && rb_r >= rb_u && ra_u > 2.0f,
            "ranges ahead / broadside: unripe %.1f / %.1f cm, ripe %.1f / %.1f cm (5 cm class cut from the sensor ring)",
            ra_u, rb_u, ra_r, rb_r);
        destroy(&h);
        dict_clear(&kw);
    }
}

// U8: taste from body size, the metadata label, the -2 draw.
static void test_ah_taste(void) {
    printf("-- AH U8: taste from body size (taste_n_a %d of %d), metadata label, the -2 draw\n", AH_FISH / 2, AH_FISH);
    Dict kw = {0};
    kw_ah(&kw, AH_FISH, AH_FISH / 2);
    Harness h = make(&kw, AH_FISH);
    Env* env = h.env;
    bool bands = true;
    bool tastes = true;
    bool same_ok = true;
    bool cross_ok = true;
    int undetected = 0;
    int n_same = 0;
    int n_cross = 0;
    for (int ep = 0; ep < 25; ep++) {
        if (ep > 0) {
            puf_reset(env);
        }
        for (int i = 0; i < AH_FISH; i++) {
            int t = i < AH_FISH / 2 ? 0 : 1;
            float lo = t == 0 ? 0.58f : 0.38f;
            float hi = t == 0 ? 0.62f : 0.42f;
            tastes = tastes && env->fish[i].taste == t;
            bands = bands && env->fish[i].size >= lo && env->fish[i].size <= hi;
            obs_t* oi = env->agents[i].observations;
            for (int o = 0; o < AH_FISH; o++) {
                if (o == i) {
                    continue;
                }
                float v = oi[OBS_META_START + (o < i ? o : o - 1)];
                if (v == -1.0f) {
                    undetected++;
                } else if (env->fish[o].taste == t) {
                    same_ok = same_ok && fabsf(v) <= 0.09f + 1e-4f;
                    n_same++;
                } else {
                    cross_ok = cross_ok && fabsf(v) >= 0.11f - 1e-4f && fabsf(v) <= 0.29f + 1e-4f;
                    n_cross++;
                }
            }
        }
    }
    CHECK(tastes && bands, "25 resets: slots < %d taste A with sizes in [0.58, 0.62], the rest B in [0.38, 0.42]", AH_FISH / 2);
    CHECK(undetected == 0, "every emitting conspecific is knollen-detected in the 40 x 40 arena (%d undetected)", undetected);
    CHECK(same_ok, "same-group metadata in [-0.09, 0.09] (%d readings)", n_same);
    CHECK(cross_ok, "cross-group |metadata| in [0.11, 0.29] (%d readings): |metadata| > 0.10 identifies the group", n_cross);
    destroy(&h);
    dict_clear(&kw);
    // taste_n_a -1 / -2 / n all make exactly one size draw per fish: same RNG state and spawn after reset
    int modes[3] = {-1, -2, AH_FISH / 2};
    unsigned int rng_after[3];
    bool same_spawn = true;
    Vec2 pos_ref[MAX_AGENTS];
    bool inband = true;
    int n_a = 0;
    for (int m = 0; m < 3; m++) {
        Dict km = {0};
        kw_ah(&km, AH_FISH, modes[m]);
        Harness hm = make(&km, AH_FISH);
        rng_after[m] = hm.env->rng;
        for (int i = 0; i < AH_FISH; i++) {
            if (m == 0) {
                pos_ref[i] = hm.env->fish[i].pos;
            } else {
                same_spawn = same_spawn && hm.env->fish[i].pos.x == pos_ref[i].x && hm.env->fish[i].pos.y == pos_ref[i].y;
            }
            if (m == 1) {
                int t = hm.env->fish[i].taste;
                float lo = t == 0 ? 0.58f : 0.38f;
                float hi = t == 0 ? 0.62f : 0.42f;
                inband = inband && hm.env->fish[i].size >= lo && hm.env->fish[i].size <= hi;
                n_a += t == 0;
            }
        }
        destroy(&hm);
        dict_clear(&km);
    }
    CHECK(rng_after[0] == rng_after[1] && rng_after[1] == rng_after[2] && same_spawn,
        "taste_n_a -1 / -2 / %d leave the RNG in the same state after reset, identical spawns (one draw per fish in every mode)",
        AH_FISH / 2);
    CHECK(inband, "taste_n_a -2: every fish's size lies inside its own taste's band (%d A, %d B this reset)", n_a, AH_FISH - n_a);
}

// V3 trace rows and typed object events (section 5.2), written to $WEF_TEST_TRACE_DIR.
static void test_ah_trace(void) {
    const char* dir = getenv("WEF_TEST_TRACE_DIR");
    if (dir == NULL || dir[0] == '\0') {
        printf("-- AH trace: skipped (set WEF_TEST_TRACE_DIR to a writable directory)\n");
        return;
    }
    printf("-- AH trace: V3 rows and typed object events\n");
    CHECK(sizeof(WefTraceRowV3) == sizeof(WefTraceRowV2) + 7 * sizeof(int32_t),
        "WefTraceRowV3 = V2 row + 7 int32 (%zu bytes)", sizeof(WefTraceRowV3));
    setenv("WEF_TRACE_DIR", dir, 1);
    Dict kw = {0};
    kw_ah(&kw, 2, 1);
    dict_set(&kw, "episode_length", 16);
    Harness h = make(&kw, 2);
    unsetenv("WEF_TRACE_DIR");
    Env* env = h.env;
    clear_objects(env);
    for (int f = 2; f < env->num_food; f++) {  // far grid, >= 6 cm from either fish
        ah_bush(env, f, 22.0f + 2.0f * (float)((f - 2) % 9), 26.0f + 2.0f * (float)((f - 2) / 9), 0, false);
    }
    ah_bush(env, 0, 12.0f, 20.0f, 1, false);  // unripe B ahead of the A fish: planted at tick 1
    ah_bush(env, 1, 31.5f, 20.0f, 0, true);   // ripe A ahead of the B fish: eaten at tick 1
    ah_recount(env);
    env->ripen_lin = 0.5f;                    // x = 63/64 ... ripenings during the episode
    place_fish(env, 0, 10.0f, 20.0f, 0.0f);
    place_fish(env, 1, 30.0f, 20.0f, 0.0f);
    set_action(&h, 0, -8.0f, 0.0f, 1.0f, AH_UPPER);
    ah_still(&h, 1);
    puf_step(env);
    ah_still(&h, 0);
    while (env->episode == 0) {
        puf_step(env);
    }
    int ripened = (int)env->log.ripened;
    int eaten = (int)env->log.collective_food;  // the B fish re-eats its bush whenever it re-ripens (p = 0.5)
    destroy(&h);  // closes and flushes the files
    char path[4096];
    snprintf(path, sizeof(path), "%s/env_%05d.v3.bin", dir, (int)g_seed);
    FILE* fp = fopen(path, "rb");
    CHECK(fp != NULL, "trace file %s exists (.v3.bin suffix under allelo)", path);
    if (fp != NULL) {
        WefTraceRowV3 rows[32];
        size_t n = fread(rows, sizeof(WefTraceRowV3), 32, fp);
        int extra = fgetc(fp);
        fclose(fp);
        CHECK(n == 32 && extra == EOF, "16 ticks x 2 fish = 32 V3 rows (%zu read, trailing bytes: %s)", n, extra == EOF ? "none" : "yes");
        if (n == 32) {
            WefTraceRowV3* p0 = &rows[0];  // tick 1, fish 0: the planter
            WefTraceRowV3* p1 = &rows[1];  // tick 1, fish 1: the eater
            CHECK(p0->tick == 1 && p0->agent == 0 && p0->taste == 0 && p0->plant_type == 1 && p0->planted_slot == 0
                && p0->ate_type == 0 && p0->hold == 31 && p0->bite == 1,
                "planter row: taste A, plant_type 1, planted_slot 0, hold 31 (eat_cooldown above EAT_COOLDOWN_STEPS)");
            CHECK(p1->taste == 1 && p1->plant_type == 0 && p1->planted_slot == -1 && p1->ate_type == 1 && p1->ate == 1
                && p1->reward == 0.5f && p1->hold == 0,
                "eater row: taste B, ate_type 1 (A bush), reward 0.5, hold 0 (plain eat cooldown)");
            CHECK(p0->n_a == 64 && p0->food_active == p0->n_ripe && p0->food_left == p0->n_ripe,
                "n_a 64 after the conversion; the V2 food_active and V1 food_left fields carry n_ripe (%d)", p0->n_ripe);
            bool holds_ok = true;
            for (int t = 1; t < 16; t++) {
                holds_ok = holds_ok && rows[2 * t].hold == (31 - t > 0 ? 31 - t : 0);
            }
            CHECK(holds_ok, "the planter's hold field counts down 31, 30, ... across the rows");
        }
    }
    snprintf(path, sizeof(path), "%s/env_%05d.obj.bin", dir, (int)g_seed);
    fp = fopen(path, "rb");
    CHECK(fp != NULL, "object-event file %s exists", path);
    if (fp != NULL) {
        WefObjEvent ev;
        int n_reset = 0;
        int n_plant_a = 0;
        int n_ripened = 0;
        int n_eaten = 0;
        int bad_kind = 0;
        int bad_event = 0;
        int plant_idx = -1;
        while (fread(&ev, sizeof(ev), 1, fp) == 1) {
            bad_kind += ev.kind < 2 || ev.kind > 3;
            bad_event += ev.event != 0 && ev.event != 3 && ev.event != 4 && ev.event != 5 && ev.event != 6;
            n_reset += ev.event == 0;
            n_ripened += ev.event == 5;
            n_eaten += ev.event == 6;
            if (ev.event == 3) {
                n_plant_a++;
                plant_idx = ev.index;
                bad_kind += ev.kind != 2;  // planted -> A carries kind 2
            }
        }
        fclose(fp);
        CHECK(bad_kind == 0 && bad_event == 0, "every object event carries kind 2 + type and an AH event code (0/3/4/5/6): %d bad kinds, %d bad codes",
            bad_kind, bad_event);
        CHECK(n_reset == 128 && n_plant_a == 1 && plant_idx == 0 && n_eaten == eaten && eaten >= 1 && n_ripened == ripened,
            "64 reset events x 2 episodes, 1 planted->A on slot 0, %d eaten (= Log collective_food %d), %d ripened (= Log ripened %d)",
            n_eaten, eaten, n_ripened, ripened);
    }
    dict_clear(&kw);
}

// Calibration mode (E1): all fish scripted, so the whole thing runs on the CPU.
//   wef_test roles=3,3,2,2 [episodes=100] [oracle=0] [preset=harvest|allelo] [shape=conv] [num_agents=N] [key=value ...]
// Prints per-slot pellets / cleans / strip time per episode and episode-level
// waste fraction and open fraction, in the scripts/cleanup.args preset.
static int run_calibration(int argc, char** argv) {
    Dict kw = {0};
    kw_base(&kw, 4);
    bool harvest = false;
    bool allelo = false;
    bool conv = false;
    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "preset=harvest") == 0) {
            harvest = true;
        } else if (strcmp(argv[a], "preset=allelo") == 0) {
            allelo = true;
        } else if (strcmp(argv[a], "shape=conv") == 0) {
            conv = true;
        }
    }
    if (allelo) {
        kw_allelo(&kw);
        if (conv) {
            kw_allelo_conv(&kw);
        }
    } else if (harvest) {
        kw_harvest(&kw);
    } else {
        kw_cleanup(&kw);
    }
    int episodes = 100;
    const char* roles = "0,0,0,0";
    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "preset=harvest") == 0 || strcmp(argv[a], "preset=allelo") == 0
                || strcmp(argv[a], "shape=conv") == 0) {
            continue;
        }
        char key[64];
        char val[64];
        if (strpbrk(argv[a], " \t") != NULL) {
            // zsh does not word-split an unquoted $VAR: "k1=v1 k2=v2" arrives as one argv and
            // sscanf would keep only k1, silently running the default preset for the rest.
            fprintf(stderr, "argument contains whitespace (unsplit variable?): '%s'\n", argv[a]);
            return 2;
        }
        if (sscanf(argv[a], "%63[^=]=%63s", key, val) != 2) {
            fprintf(stderr, "bad arg %s\n", argv[a]);
            return 2;
        }
        if (strcmp(key, "roles") == 0) {
            roles = argv[a] + 6;
        } else if (strcmp(key, "seed") == 0) {
            g_seed = (unsigned int)atoi(val);
        } else if (strcmp(key, "episodes") == 0) {
            episodes = atoi(val);
        } else if (strcmp(key, "oracle") == 0) {
            dict_set(&kw, "bot_oracle", atof(val));
        } else {
            dict_set(&kw, key, atof(val));
        }
    }
    puf_ini_set(&kw, "roles", roles);   // parses the comma list into values[]/len
    int num_fish = (int)dict_get(&kw, "num_agents");   // num_agents=N on the command line
    Harness h = make(&kw, num_fish);
    Env* env = h.env;
    double pellets[MAX_AGENTS] = {0};
    double cleans[MAX_AGENTS] = {0};
    double strip[MAX_AGENTS] = {0};
    double ret[MAX_AGENTS] = {0};      // raw (taste-weighted) return per slot
    double plant_a[MAX_AGENTS] = {0};  // conversions to A / B per slot
    double plant_b[MAX_AGENTS] = {0};
    double noop[MAX_AGENTS] = {0};
    double zaps_given[MAX_AGENTS] = {0};
    int taste_of[MAX_AGENTS] = {0};
    double waste_frac = 0.0;
    double open_frac = 0.0;
    double collective = 0.0;
    double stock_left = 0.0;
    double collapsed = 0.0;
    double survival = 0.0;
    double mono_final = 0.0;
    double mono_mean = 0.0;
    double t_conv = 0.0;
    double plantings = 0.0;
    double proactive = 0.0;
    double zaps_cross = 0.0;
    double zaps_same = 0.0;
    double zaps_taken[MAX_AGENTS] = {0};
    double n_ripe_mean = 0.0;
    double plant_noop_total = 0.0;
    double plant_attempts = 0.0;
    double zaps_given_step[MAX_AGENTS] = {0};  // per-episode accumulators (reset each episode)
    double zaps_taken_step[MAX_AGENTS] = {0};
    double plant_a_step[MAX_AGENTS] = {0};
    double plant_b_step[MAX_AGENTS] = {0};
    double noop_step[MAX_AGENTS] = {0};
    const char* dbg = getenv("WEF_TEST_DEBUG");
    for (int e = 0; e < episodes; e++) {
        for (int i = 0; i < env->num_agents; i++) {
            zaps_given_step[i] = 0;
            zaps_taken_step[i] = 0;
            plant_a_step[i] = 0;
            plant_b_step[i] = 0;
            noop_step[i] = 0;
        }
        while (env->episode == e) {
            puf_step(env);
            if (env->tick > 0) {
                // per-slot event tallies from this step's trace fields (the env keeps totals only)
                for (int i = 0; i < env->num_agents; i++) {
                    FishAgent* f = &env->fish[i];
                    zaps_given_step[i] += f->bite_victim >= 0;
                    zaps_taken_step[i] += f->was_bitten;
                    plant_a_step[i] += f->trace_plant_type == 1;
                    plant_b_step[i] += f->trace_plant_type == 2;
                    noop_step[i] += f->trace_plant_noop;  // per-fish same-type-only no-op flag (exact)
                }
                n_ripe_mean += (double)env->n_ripe / env->cur_episode_length / episodes;
            }
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
                collapsed += env->collapsed ? 1.0 : 0.0;
                survival += env->collapsed ? (double)env->collapse_tick / env->cur_episode_length : 1.0;
                waste_frac += env->waste_frac_sum / env->cur_episode_length;
                open_frac += (double)env->open_steps / env->cur_episode_length;
                for (int i = 0; i < env->num_agents; i++) {
                    ret[i] += env->return_by[i];
                    taste_of[i] = env->fish[i].taste;
                    plant_a[i] += plant_a_step[i];
                    plant_b[i] += plant_b_step[i];
                    noop[i] += noop_step[i];
                    zaps_given[i] += zaps_given_step[i];
                    zaps_taken[i] += zaps_taken_step[i];
                }
                int n_max = env->n_type[0] > env->n_type[1] ? env->n_type[0] : env->n_type[1];
                mono_final += (double)n_max / env->num_food;
                mono_mean += env->mono_frac_sum / env->cur_episode_length;
                t_conv += env->convention_tick > 0 ? (double)env->convention_tick / env->cur_episode_length : 1.0;
                plantings += env->plantings;
                proactive += env->plant_proactive;
                zaps_cross += env->zaps_cross;
                zaps_same += env->zaps_same;
                plant_noop_total += env->plant_noop;
                plant_attempts += env->plant_attempts;
            }
        }
    }
    printf("roles=%s oracle=%d episodes=%d\n", roles, env->bot_oracle, episodes);
    printf("  collective pellets/episode %.1f   waste_frac %.3f   open_frac %.3f   stock_left %.1f   collapsed %.2f   survival %.3f\n",
        collective / episodes, waste_frac / episodes, open_frac / episodes, stock_left / episodes,
        collapsed / episodes, survival / episodes);
    if (env->allelo) {
        printf("  mono_final %.3f   mono_mean %.3f   t_conv %.3f   plantings %.1f   plant_proactive %.1f   plant_noop %.1f   plant_attempts %.1f   zaps_cross %.1f   zaps_same %.1f   n_ripe_mean %.2f\n",
            mono_final / episodes, mono_mean / episodes, t_conv / episodes, plantings / episodes,
            proactive / episodes, plant_noop_total / episodes, plant_attempts / episodes,
            zaps_cross / episodes, zaps_same / episodes, n_ripe_mean);
        printf("  slot role taste pellets  return plant_a plant_b plant_noop zaps_given zaps_taken\n");
        for (int i = 0; i < env->num_agents; i++) {
            printf("  %4d %4d %5c %7.1f %7.2f %7.1f %7.1f %10.1f %10.1f %10.1f\n", i, env->roles[i],
                taste_of[i] == 0 ? 'A' : 'B', pellets[i] / episodes, ret[i] / episodes,
                plant_a[i] / episodes, plant_b[i] / episodes, noop[i] / episodes,
                zaps_given[i] / episodes, zaps_taken[i] / episodes);
        }
    } else {
        printf("  slot role pellets cleans strip%%  cleans/strip-step\n");
        for (int i = 0; i < env->num_agents; i++) {
            double strip_steps = strip[i] / episodes * env->episode_length;
            printf("  %4d %4d %7.1f %6.1f %5.1f%%  %.3f\n", i, env->roles[i], pellets[i] / episodes,
                cleans[i] / episodes, 100.0 * strip[i] / episodes,
                strip_steps > 0 ? cleans[i] / episodes / strip_steps : 0.0);
        }
    }
    puf_close(env);
    dict_clear(&kw);
    return 0;
}

// Bench mode (CPU, single thread): wef_test bench [steps=N] [preset=harvest|allelo] [shape=conv] [seed=N] [num_agents=N] [project=K] [key=value ...]
// Steps one env with deterministic pseudo-random actions in the training default config
// (4 fish, 30-400 cm arenas, 64 pellets, 512-step episodes) and prints microseconds per
// env-step and per agent-step, the share spent in compute_observations, and an FNV-1a hash of
// every observation and reward produced: two builds that print the same hash are bit-identical
// on this trajectory (the check to run after any wef.h or compile-flag change).
// project=K (U12, design 4.2): also hashes the K-fish layout projection of every obs row (the
// first K - 1 knollen blocks and metadata slots; identity when K == MAX_AGENTS) plus the
// rewards, and the rewards alone. The 8-fish build at num_agents=4 project=4 must print the
// 4-fish build's proj_hash and rew_hash.
static uint64_t fnv1a(uint64_t h, const void* data, size_t n) {
    const unsigned char* p = (const unsigned char*)data;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static uint32_t g_xs = 0x9E3779B9u;
static float xs_uniform(void) {  // xorshift32 -> [0, 1)
    g_xs ^= g_xs << 13; g_xs ^= g_xs >> 17; g_xs ^= g_xs << 5;
    return (float)(g_xs >> 8) * (1.0f / 16777216.0f);
}

static int run_bench(int argc, char** argv) {
    Dict kw = {0};
    kw_base(&kw, 4);
    // training defaults from config/wef.ini
    dict_set(&kw, "min_arena_width", 30);
    dict_set(&kw, "max_arena_width", 400);
    dict_set(&kw, "min_arena_height", 30);
    dict_set(&kw, "max_arena_height", 400);
    dict_set(&kw, "food_distribution", FOOD_RANDOM);
    dict_set(&kw, "episode_length", 512);
    long steps = 20000;
    int project = 0;
    for (int a = 2; a < argc; a++) {
        if (strcmp(argv[a], "preset=harvest") == 0) {
            kw_harvest(&kw);
            continue;
        }
        if (strcmp(argv[a], "preset=allelo") == 0) {
            kw_allelo(&kw);  // 8 fish: needs the -DMAX_AGENTS=8 build (num_agents=4 projects to 4)
            continue;
        }
        if (strcmp(argv[a], "shape=conv") == 0) {
            kw_allelo_conv(&kw);
            continue;
        }
        char key[64];
        char val[64];
        if (strpbrk(argv[a], " \t") != NULL || sscanf(argv[a], "%63[^=]=%63s", key, val) != 2) {
            fprintf(stderr, "bad arg %s\n", argv[a]);
            return 2;
        }
        if (strcmp(key, "steps") == 0) {
            steps = atol(val);
        } else if (strcmp(key, "seed") == 0) {
            g_seed = (unsigned int)atoi(val);
        } else if (strcmp(key, "project") == 0) {
            project = atoi(val);
        } else {
            dict_set(&kw, key, atof(val));
        }
    }
    Harness h = make(&kw, (int)dict_get(&kw, "num_agents"));   // num_agents=N on the command line
    Env* env = h.env;
    int n = env->num_agents;
    if (project > 0 && (project > MAX_AGENTS || n > project)) {
        fprintf(stderr, "project=%d needs num_agents (%d) <= K <= MAX_AGENTS (%d)\n", project, n, MAX_AGENTS);
        return 2;
    }
    uint64_t hash = 1469598103934665603ULL;
    uint64_t proj_hash = 1469598103934665603ULL;
    uint64_t rew_hash = 1469598103934665603ULL;
    double food_sum = 0.0;
    double t0 = now_s();
    for (long s = 0; s < steps; s++) {
        for (int i = 0; i < n; i++) {
            set_action(&h, i, 6.0f * xs_uniform() - 3.0f, 4.0f * xs_uniform() - 2.0f,
                xs_uniform() < 0.86f ? 1.0f : -1.0f, xs_uniform() < 0.05f ? 1.0f : -1.0f);
        }
        puf_step(env);
        hash = fnv1a(hash, h.obs, (size_t)n * OBS_SIZE * sizeof(obs_t));
        hash = fnv1a(hash, h.rew, (size_t)n * sizeof(float));
        if (project > 0) {
            // K-fish layout: morm + amp, the first K - 1 knollen blocks (contiguous from slot 60),
            // the first K - 1 metadata slots, then last action + 7 scalars. Valid conspecifics
            // fill the first num_agents - 1 slots in both builds, so at num_agents <= K the
            // projection of the big build equals the K-fish build's row.
            size_t blocks = (size_t)(project - 1);
            for (int i = 0; i < n; i++) {
                const obs_t* row = h.obs + (size_t)i * OBS_SIZE;
                proj_hash = fnv1a(proj_hash, row, (NUM_MORMYROMASTS + NUM_AMPULLARY) * sizeof(obs_t));
                proj_hash = fnv1a(proj_hash, row + NUM_MORMYROMASTS + NUM_AMPULLARY, blocks * NUM_KNOLLEN * sizeof(obs_t));
                proj_hash = fnv1a(proj_hash, row + OBS_META_START, blocks * sizeof(obs_t));
                proj_hash = fnv1a(proj_hash, row + OBS_ACT_START, (ACTION_SIZE + 7) * sizeof(obs_t));
            }
            proj_hash = fnv1a(proj_hash, h.rew, (size_t)n * sizeof(float));
            rew_hash = fnv1a(rew_hash, h.rew, (size_t)n * sizeof(float));
        }
        food_sum += env->allelo ? env->n_ripe : env->food_active;
    }
    double t_step = now_s() - t0;
    // compute_observations alone (extra calls; they only advance the sensor-noise RNG)
    long obs_steps = steps / 4 > 0 ? steps / 4 : 1;
    double t_obs = 0.0;
    for (long s = 0; s < obs_steps; s++) {
        for (int i = 0; i < n; i++) {
            set_action(&h, i, 6.0f * xs_uniform() - 3.0f, 4.0f * xs_uniform() - 2.0f,
                xs_uniform() < 0.86f ? 1.0f : -1.0f, xs_uniform() < 0.05f ? 1.0f : -1.0f);
        }
        puf_step(env);
        double a = now_s();
        compute_observations(env);
        t_obs += now_s() - a;
    }
    double us_step = 1e6 * t_step / (double)steps;
    double us_obs = 1e6 * t_obs / (double)obs_steps;
    printf("bench steps=%ld fish=%d mean_%s=%.1f\n", steps, n, env->allelo ? "n_ripe" : "food_active", food_sum / (double)steps);
    printf("  %.2f us/env-step  %.2f us/agent-step  %.0f agent-steps/s (1 thread)\n",
        us_step, us_step / n, 1e6 * n / us_step);
    printf("  compute_observations %.2f us/env-step (%.0f%% of step)\n", us_obs, 100.0 * us_obs / us_step);
    printf("  hash %016llx\n", (unsigned long long)hash);
    if (project > 0) {
        printf("  proj_hash %016llx (%d-fish layout, %d slots + rewards)  rew_hash %016llx\n",
            (unsigned long long)proj_hash, project, 71 + 13 * (project - 1), (unsigned long long)rew_hash);
        if (project == MAX_AGENTS && proj_hash != hash) {
            printf("  FAIL: the identity projection (K = MAX_AGENTS) must reproduce the full hash\n");
            return 1;
        }
    }
    puf_close(env);
    dict_clear(&kw);
    return 0;
}

// The inlined glibc rand_r must produce the library's stream (env RNG bit-identity).
static void test_rand_r(void) {
    unsigned int a = 12345u, b = 12345u;
    int same = 1;
    for (int i = 0; i < 100000 && same; i++) {
        same = wef_rand_r(&a) == rand_r(&b) && a == b;
    }
    CHECK(same, "wef_rand_r matches glibc rand_r over 100000 draws");
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        return run_bench(argc, argv);
    }
    if (argc > 1) {
        return run_calibration(argc, argv);
    }
    test_rand_r();
    test_sensing();
    test_dynamics();
    test_actions();
    test_harvest();
    test_commons();
    test_seasonal();
    test_patchy_commons();
    test_baseline();
    // Allelopathic Harvest (design 7.1 U1-U10, the Log and the trace); U11 / U12 in run_test.sh
    test_ah_layout();
    test_ah_ripening();
    test_ah_eating();
    test_ah_planting();
    test_ah_log();
    test_ah_sensing_type();
    test_ah_sensing_ripe();
    test_ah_confound();
    test_ah_taste();
    test_ah_trace();
    printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
