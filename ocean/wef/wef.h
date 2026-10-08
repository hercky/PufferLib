// Weakly electric fish env. Port of KempnerInstitute/wef biophysics.
// Positions in cm; field measure converts to m and returns V/m.

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"
#include "rlgl.h"  // rlReadScreenPixels / rlDrawRenderBatchActive (WEF_VIDEO_OUT capture)
typedef float obs_t;
#include "pufferenv.h"

// 5c trainer macros (no binding.c). Float obs; pufferl converts on upload.
#define ACT_SIZES {1, 1, 1, 1}
#define NUM_ATNS 4

// Constants
#define PI_F 3.14159265358979323846f
#define CM_TO_M 0.01f
#define K_COULOMB 8.99e9f
#define EPSILON_0 8.854e-12f
#define FIELD_EPS_M 1e-5f
#define SENSOR_EPS 1e-25f

// Values from upstream cfg.py and Table 1 of the accompanying publication
#define SIMULATION_HZ 83.0f
#define BODY_RADIUS_CM 1.0f
#define FOOD_RADIUS_CM 0.25f
#define CONDUCTOR_CONTRAST -0.5f
#define EOD_CHARGE_C 1.11e-15f
#define EOD_POLE_OFFSET_CM 0.5f
#define INTRINSIC_MOMENT_C_M 1.11e-23f
#define FOOD_INTRINSIC_MOMENT_C_M 1.11e-24f

// Max speeds: 35 cm/s * 2, 3.6 rad/s * 4; then / SIMULATION_HZ.
#define MAX_LINEAR_VELOCITY_CM_S 70.0f
#define MAX_ANGULAR_VELOCITY_RAD_S 14.4f
#define SIZE_SPEED_EXPONENT 1.0f

// Sensors
#define NUM_MORMYROMASTS 36
#define NUM_AMPULLARY 24
#define NUM_KNOLLEN 12
// Fish per arena. The 8-fish Allelopathic Harvest build passes -DMAX_AGENTS=8 (a separate
// binary: the knollenorgan block and the metadata slots scale with MAX_AGENTS - 1).
#ifndef MAX_AGENTS
#define MAX_AGENTS 4
#endif

#define MAX_FOOD 64
#define ACTION_SIZE 4
// morm + amp + (knollen + metadata) per other fish + last action + 7 scalars:
// 110 at MAX_AGENTS 4 (upstream), 162 at 8.
#define OBS_BASE_SIZE (NUM_MORMYROMASTS + NUM_AMPULLARY + (NUM_KNOLLEN + 1) * (MAX_AGENTS - 1) + ACTION_SIZE + 7)
// Institution build (-DWEF_INST, docs/institutions-plan.md section 4): the row grows by one rule
// slot, one own-mark slot and one mark slot per other fish (114 at 4 fish, 170 at 8). The default
// and the 8-fish AH builds add nothing, so their rows and checkpoints are untouched.
#ifdef WEF_INST
#define INST_OBS_EXTRA (2 + (MAX_AGENTS - 1))
#else
#define INST_OBS_EXTRA 0
#endif
#define OBS_SIZE (OBS_BASE_SIZE + INST_OBS_EXTRA)
#define EATING_RADIUS_CM 2.0f
#define BITING_RADIUS_CM 3.0f
#define EATING_ANGLE (PI_F / 4.0f)
#define MAX_PATCHES 90
#define MAX_WASTE 64

// Reward coefficients
#define EAT_REWARD 1.0f
#define PROXIMITY_SHAPING_REWARD 0.1f
#define BITTEN_REWARD -0.5f
#define BITE_REWARD -0.0001f
#define COLLISION_REWARD -0.05f
#define EFFORT_OVER_REWARD -0.01f
#define PENALIZE_EFFORT_OVER_FRAC 0.5f

#define TRACE_LENGTH 65

#define EAT_COOLDOWN_STEPS 3
#define BITE_COOLDOWN_STEPS 5

#define AMPULLARY_MIN_VM 2e-10f
#define AMPULLARY_MAX_VM 2e-8f
#define MORMYROMAST_MIN_VM 5e-8f
#define MORMYROMAST_MAX_VM 5e-2f
#define KNOLLEN_MIN_VM 2e-7f

// Sensor interaction ranges in cm.
// Food/prey and conspecifics (other agents) have separate cutoffs for morm & amp.
#define MORM_FOOD_RANGE_CM 5.0f       // prey ≤ 5 cm
#define MORM_AGENT_RANGE_CM 10.0f     // agents ≤ 10 cm
#define AMP_FOOD_RANGE_CM 4.0f        // prey ≤ 4 cm
#define AMP_AGENT_RANGE_CM 8.0f       // conspecifics ≤ 8 cm
#define KNOLLEN_AGENT_RANGE_CM 100.0f // conspecific EOD ≤ 100 cm

// Arena wall indices for first-order image charges.
#define WALL_LEFT 0
#define WALL_RIGHT 1
#define WALL_BOTTOM 2
#define WALL_TOP 3
#define NUM_WALLS 4

// Render palette
#define WEF_COLOR_BG            ((Color){6, 24, 24, 255})
#define WEF_COLOR_MIDGRAY       ((Color){120, 120, 120, 255})
#define WEF_COLOR_PANEL         ((Color){10, 10, 10, 200})
#define WEF_COLOR_TEXT          ((Color){240, 240, 240, 255})
#define WEF_COLOR_FISH          ((Color){180, 150, 230, 255})
#define WEF_COLOR_FISH_PULSE    ((Color){180, 150, 230, 140})
#define WEF_COLOR_SENSOR        ((Color){184, 164, 224, 200})
#define WEF_COLOR_FOOD          ((Color){0x7A, 0xEF, 0x9A, 200})  // lighter green (AH: type A)
#define WEF_COLOR_FOOD_B        ((Color){0xF0, 0x7A, 0xD0, 200})  // AH: type B bushes (magenta)
#define WEF_COLOR_HOLD          ((Color){250, 200, 60, 255})     // AH: planter on hold (body fill)
#define WEF_COLOR_WASTE         ((Color){235, 150, 60, 220})     // inert debris (cleanup mode)
#define WEF_COLOR_INST          ((Color){255, 225, 120, 255})    // institution beacon (inst_obs)
#define WEF_COLOR_MARK          ((Color){255, 255, 255, 255})    // violation mark on a fish (white ring, dark outline)
#define WEF_COLOR_EOD_POS       ((Color){220, 60, 50, 255})
#define WEF_COLOR_EOD_NEG       ((Color){60, 120, 255, 255})
#define WEF_COLOR_BITE          ((Color){255, 70, 70, 255})
#define WEF_BITE_FLASH_FRAMES   30

// Bite-selectivity metrics (Log): a fish is a "recent eater" for WEF_RECENT_STEPS after a
// pellet; a "defector" if it recently ate while the stock was at or below sustain_frac * K
// (config key, default 0.5; regrowth modes) or ate without cleaning for WEF_CLEAN_MEMORY
// steps (Cleanup).
#define WEF_RECENT_STEPS 32
#define WEF_CLEAN_MEMORY 256

// Field arrows: yellow (min) → red (max) log gradient over WEF |E| in V/cm.
#define WEF_FIELD_LOG_LO        (-7.0f)   // 1e-7 V/cm
#define WEF_FIELD_LOG_HI        (-3.0f)   // 1e-3 V/cm
#define WEF_COLOR_FIELD_WEAK    ((Color){255, 255, 40, 200})   // yellow (min |E|)
#define WEF_COLOR_FIELD_STRONG  ((Color){230, 40, 30, 255})    // red (max |E|)

typedef struct { float x, y; } Vec2;

// One electroreceptor site in the fish body frame (AoS).
typedef struct Sensor {
    Vec2 p;  // position on body
    Vec2 n;  // outward normal
} Sensor;

// Per-fish dynamics / EOD / dipoles. Sensor geometry is shared (see g_*).
typedef struct FishAgent {
    Vec2 pos;
    float orientation;
    float max_linear_velocity;
    float max_angular_velocity;
    Vec2 disp_ego;
    float size;
    int bite_cooldown;
    int eat_cooldown;
    bool emits_eod;
    bool bite_action;
    bool was_bitten;
    bool ate;         // trace only: ate a pellet this step
    bool collided;    // trace only: move reverted by a fish collision this step
    int bite_victim;  // trace only: index of fish bitten this step, -1 if none
    int freeze;       // cleanup: steps left of the bitten timeout (0 unless bitten_freeze_steps > 0)
    bool can_clean;   // cleanup: false for enforced free-riders (no_clean_agents)
    int cleaned;      // trace only: waste items removed this step
    float trace_nearest_food;  // trace only: nearest active pellet this step, -1 if none
    int bot_dir;      // scripted roles: patrol direction (+1 / -1)
    int eat_mark;     // metrics: tick of the last pellet eaten (0 = none this episode)
    int low_eat_mark; // metrics: tick of the last pellet eaten with stock <= sustain_frac * K
    int clean_mark;   // metrics: tick of the last cleaning bite
    Vec2 bot_last;    // scripted roles: position at the previous decision (stall detection)
    int bot_stall;    // scripted roles: consecutive decisions without moving
    int bot_escape;   // scripted roles: steps left of a random-heading escape
    float bot_escape_turn;
    Vec2 bot_last_eat;  // scripted roles (bot_camp): where this fish last ate
    bool bot_has_eat;
    bool has_previous_food_distance;
    float previous_food_distance;
    float last_action[ACTION_SIZE];
    Vec2 eod_pos[2];
    float eod_charge[2];
    Vec2 intrinsic_moment;
    Vec2 induced_moment;
    // --- Allelopathic Harvest (allelo). All zero / unread on the default path.
    int taste;             // 0 = A, 1 = B (set at reset from the size draw)
    bool bite_upper;       // this step's bite magnitude landed in the upper bin (raw[3] > plant_split)
    int last_plant_type;   // type of this fish's last conversion, -1 none
    int plant_mark;        // tick of the last conversion (0 = none this episode; defector metric)
    int last_eaten_slot;   // slot of the last bush eaten, -1 none (plant_proactive)
    int last_eat_tick;
    int trace_plant_type;  // trace only: 0 none / 1 A / 2 B converted this step
    int trace_planted_slot;// trace only: slot converted this step, -1 none
    int trace_ate_type;    // trace only: 0 none / 1 A / 2 B eaten this step
    bool trace_plant_noop; // trace only: this step's bite was a same-type-only no-op (plant_noop)
    // --- Institutions (inst_mode / inst_obs, docs/institutions-plan.md 3.2-4). Zero / unread when
    // both are 0.
    int mark;              // steps left of the violation mark (the public label)
    float inst_val;        // the rule as last read at the beacon: +1 / -1, 0 = never read (latched)
    bool informed;         // has read the beacon this episode
    int first_visit_tick;  // first tick inside inst_read_cm, 0 = never
    int last_read_tick;    // last tick inside inst_read_cm (scripted compliers re-read the season)
    int visits;            // fish-steps inside inst_read_cm
    int inst_private;      // inst_mode 2 (AH): this fish's private prescribed type 0 / 1
    float inst_private_theta;  // inst_mode 2 (Commons): this fish's private closing threshold
    int violations;        // rule-breaking acts this episode
    int comply_acts;       // rule-governed acts that followed the rule
    int rule_acts;         // rule-governed acts (AH: conversions; Commons: eats)
    bool trace_violated;   // trace only: this step's act broke the rule
    bool trace_zap_marked; // trace only: this step's zap landed on a marked fish
} FishAgent;

// Shared sensor layouts (body radius fixed → identical for every fish)
Sensor g_morm[NUM_MORMYROMASTS];
Sensor g_amp[NUM_AMPULLARY];
Sensor g_knollen[NUM_KNOLLEN];

float clamp(float value, float minimum, float maximum) {
    return fminf(maximum, fmaxf(minimum, value));
}

float wrap_angle(float angle) {
    return atan2f(sinf(angle), cosf(angle));
}

// Log-scale electroreceptor encoding used by mormyromast + ampullary.
float encode_log_sensor(float reading, float lo, float hi) {
    if (reading == 0.0f) {
        return 0.0f;
    }
    float sign = reading < 0.0f ? -1.0f : 1.0f;
    float mag = fmaxf(clamp(fabsf(reading), lo, hi), SENSOR_EPS);
    float nrm = (log10f(mag) - log10f(lo)) / (log10f(hi) - log10f(lo));
    return sign * clamp(nrm, 0.0f, 1.0f);
}

// Sensor local frame → world pose of fish
Sensor sensor_world(const Sensor* s, const FishAgent* fish) {
    float c = cosf(fish->orientation);
    float sn = sinf(fish->orientation);
    return (Sensor){
        {
            c * s->p.x - sn * s->p.y + fish->pos.x,
            sn * s->p.x + c * s->p.y + fish->pos.y,
        },
        {
            c * s->n.x - sn * s->n.y,
            sn * s->n.x + c * s->n.y,
        },
    };
}

// Same as sensor_world with the fish's cos/sin passed in (hoisted out of the sensor loops;
// the arithmetic is identical, so observations are bit-identical).
static inline Sensor sensor_world_cs(const Sensor* s, const FishAgent* fish, float c, float sn) {
    return (Sensor){
        {
            c * s->p.x - sn * s->p.y + fish->pos.x,
            sn * s->p.x + c * s->p.y + fish->pos.y,
        },
        {
            c * s->n.x - sn * s->n.y,
            sn * s->n.x + c * s->n.y,
        },
    };
}

static inline bool in_forward_cone(const FishAgent* fish, Vec2 target,
        float radius_cm, float cone) {
    float dx = target.x - fish->pos.x;
    float dy = target.y - fish->pos.y;
    if (dx * dx + dy * dy >= radius_cm * radius_cm) {
        return false;
    }
    float bearing = atan2f(dy, dx);
    return fabsf(wrap_angle(bearing - fish->orientation)) <= cone * 0.5f;
}

// Induced dipole scale for a conducting sphere of radius_cm (contrast κ).
float conductor_scale(float radius_cm) {
    float r = radius_cm * CM_TO_M;
    return 3.0f * EPSILON_0 * CONDUCTOR_CONTRAST *
        (4.0f / 3.0f) * PI_F * r * r * r;
}

// Same as conductor_scale with an explicit contrast (waste objects). Kept separate
// so the baseline call sites and their constant folding are untouched.
float conductor_scale_c(float radius_cm, float contrast) {
    float r = radius_cm * CM_TO_M;
    return 3.0f * EPSILON_0 * contrast * (4.0f / 3.0f) * PI_F * r * r * r;
}

// Caps: 2 EOD poles/agent; agent+food+waste dipoles.
enum { WEF_MAX_MONO = 2 * MAX_AGENTS, WEF_MAX_DIP = MAX_AGENTS + MAX_FOOD + MAX_WASTE + 1 };  // + 1 beacon

// Electric sources for measure_field: position in meters, moments/charges SI.
typedef struct { Vec2 p; float q; } Mono;
typedef struct { Vec2 p, m; } Dipole;

Vec2 to_m(Vec2 p_cm) {
    return (Vec2){p_cm.x * CM_TO_M, p_cm.y * CM_TO_M};
}

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float food_eaten_mean;
    float eod_rate;
    float collisions_fish;
    float bites;
    float food_per_fish_area;
    // Cleanup-mode metrics (docs/wef-cleanup-v0-design.md section 9). Per-episode means.
    float collective_food;   // pellets eaten by all fish
    float waste_frac;        // time-mean of waste_active / waste_max
    float frac_open;         // fraction of steps with regrowth probability > 0
    float cleans;            // waste items removed by fish (Hughes' public-good contribution)
    float regrown;           // pellets spawned by regrowth
    float equality;          // 1 - Gini of pellets eaten per fish
    float clean_gini;        // Gini of cleaning contributions
    float clean_max_share;   // largest single fish share of cleaning
    float strip_frac;        // fraction of fish-steps spent in the waste strip
    float bites_in_strip;
    float freezes;
    // Bite selectivity (enforcement signal). Selectivity toward defectors =
    // (bites_on_defectors / bites) / defector_frac; > 1 means bites target defectors.
    float bites_on_eaters;   // victim ate within WEF_RECENT_STEPS
    float bites_on_defectors;// victim is a "defector" (see sustain_frac / WEF_CLEAN_MEMORY)
    float bites_at_risk;     // bite while the commons is at risk (stock <= K/2; Cleanup: quality < 0.5)
    float bites_top;         // bites by the single most-biting fish (share = bites_top / bites)
    float eater_frac;        // time-mean fraction of fish that are recent eaters
    float defector_frac;     // time-mean fraction of fish that are defectors
    float frozen_frac;       // fraction of fish-steps spent frozen (bitten_freeze_steps)
    // Commons collapse (GovSim-style): regrowth modes 2 and 3.
    float collapsed;         // fraction of episodes in which the stock collapsed
    float survival_frac;     // collapse tick / episode length (1 if it never collapsed)
    float policy_0_score;    // mean raw return of fish on policy 0 (match eval)
    float policy_1_score;
    float draw_rate;         // always 0; required by match eval
    // Allelopathic Harvest (docs/wef-allelopathic-harvest-v0-design.md 5.1); zero unless allelo.
    float mono_frac;         // time-mean max(n_A, n_B) / num_food (the paper's m-bar; AH perf)
    float mono_final;        // max(frac_a_final, 1 - frac_a_final)
    float frac_a_final;      // n_A / num_food at the last tick
    float frac_a_mean;       // time-mean n_A / num_food (signed composition)
    float majority_frac_final; // fraction of bushes of the majority taste's type at the end; 0.5 on a tie
    float conv_c;            // |frac_a_mean - g_A|, g_A = fraction of A-tasting fish (conventionality)
    float time_to_convention;// first tick with max(n_A, n_B) >= conv_theta * num_food, / T; 1 if never
    float ripened;           // ripening events
    float ripe_frac;         // time-mean n_ripe / num_food
    float plantings;         // conversions
    float plantings_a;       // conversions by A-tasting planters
    float plantings_b;
    float plant_attempts;    // AH-mode bites that found no fish (conversions + no-ops + misses)
    float plant_own_frac;    // conversions to the planter's own type / plantings
    float plant_own_frac_a;
    float plant_own_frac_b;
    float plant_major_frac;  // conversions to the strictly more common type / plantings
    float plant_proactive;   // conversions of a bush the planter did not eat within the previous 3 steps
    float plant_noop;        // same-type-only planting bites
    float plant_gini;        // Gini of plantings per fish
    float eaten_match_frac;  // eats of the fish's own type / eats
    float zaps_cross;        // fish bites with attacker taste != victim taste
    float zaps_same;
    float taste_a_return;    // mean raw return of A-tasting fish (0 if none)
    float taste_b_return;
    float taste_a_n;         // number of A-tasting fish
    // Institutions (docs/institutions-plan.md 4-5); zero unless inst_mode > 0 or inst_obs.
    float inst_type_a;       // AH: fraction of episodes whose prescribed type at reset was A
    float comply_frac;       // rule-following acts / rule-governed acts (1 when there were none)
    float comply_informed;   // the same over fish that had read the beacon
    float violations;        // rule-breaking acts per episode
    float marked_frac;       // fish-steps marked / fish-steps
    float marked_exposure;   // (i, j) pairs within MORM_AGENT_RANGE_CM with j marked / all such pairs (the mark-blind null)
    float zaps_on_marked;    // fish bites whose victim was marked
    float zaps_on_marked_share;  // zaps_on_marked / bites (0 without bites)
    float inst_agree_final;  // AH: bushes of the prescribed type / K at the end
    float inst_agree_mean;   // AH: time-mean of the same
    float plantings_p;       // AH: conversions by fish whose taste is the prescribed type
    float plantings_np;      // AH: conversions by the other group (= violations under plant_mode 2)
    float closed_frac;       // Commons: steps with the public season closed / T
    float eats_closed;       // Commons: eats that broke the rule
    float mark_zap_bounty;   // bounty paid for zapping marked fish
    float beacon_first_visit;// mean over fish of the first-read tick / T (1 if never)
    float informed_frac;     // fish that read the beacon at least once / N
    float informed_mean;     // time-mean fraction of informed fish
    float beacon_visits;     // fish-steps inside inst_read_cm / N
    float n;
};

// Optional trajectory export for offline analysis. When the WEF_TRACE_DIR env var
// is set, every env appends one WefTraceRow per (step, fish) to
// $WEF_TRACE_DIR/env_<id>.bin. Unset → no I/O, and no effect on dynamics or RNG.
// All fields are 4 bytes so the file loads as a flat numpy structured array.
typedef struct WefTraceRow {
    int32_t env_id, episode, tick, agent;
    float x, y, orientation, size;
    float move, turn;
    int32_t eod, bite, bite_victim, was_bitten, ate, collided;
    float reward, nearest_food, arena_x, arena_y;
    int32_t food_left;
} WefTraceRow;

// Cleanup-mode trace row (file name env_<id>.v2.bin): WefTraceRow + cleanup state.
typedef struct WefTraceRowV2 {
    int32_t env_id, episode, tick, agent;
    float x, y, orientation, size;
    float move, turn;
    int32_t eod, bite, bite_victim, was_bitten, ate, collided;
    float reward, nearest_food, arena_x, arena_y;
    int32_t food_left;
    int32_t cleaned, waste_left, food_active, frozen, zone;  // zone: 0 mid, 1 strip, 2 orchard
} WefTraceRowV2;

// Allelopathic Harvest trace row (file name env_<id>.v3.bin): WefTraceRowV2 + AH state. In AH
// mode the V2 food_active field and the V1 food_left field both carry n_ripe (bushes are
// permanent and eaten repeatedly, so num_food - food_eaten would go negative).
typedef struct WefTraceRowV3 {
    int32_t env_id, episode, tick, agent;
    float x, y, orientation, size;
    float move, turn;
    int32_t eod, bite, bite_victim, was_bitten, ate, collided;
    float reward, nearest_food, arena_x, arena_y;
    int32_t food_left;
    int32_t cleaned, waste_left, food_active, frozen, zone;
    int32_t taste;         // 0 A, 1 B
    int32_t plant_type;    // 0 none / 1 A / 2 B converted this step
    int32_t planted_slot;  // -1 none
    int32_t ate_type;      // 0 none / 1 A / 2 B eaten this step
    int32_t n_a, n_ripe;
    int32_t hold;          // eat_cooldown while above EAT_COOLDOWN_STEPS (a planting hold), else 0
} WefTraceRowV3;

// Institution trace row (file name env_<id>.v4.bin, written when inst_mode > 0 or inst_obs):
// WefTraceRowV3 + the rule as this fish has read it, its mark, this step's violation / marked zap,
// whether it has read the beacon, and the public rule state (AH: 0 A / 1 B; Commons: 1 closed).
typedef struct WefTraceRowV4 {
    int32_t env_id, episode, tick, agent;
    float x, y, orientation, size;
    float move, turn;
    int32_t eod, bite, bite_victim, was_bitten, ate, collided;
    float reward, nearest_food, arena_x, arena_y;
    int32_t food_left;
    int32_t cleaned, waste_left, food_active, frozen, zone;
    int32_t taste, plant_type, planted_slot, ate_type, n_a, n_ripe, hold;
    int32_t inst_signal;   // -1 / 0 (never read) / +1
    int32_t mark;          // steps left of this fish's mark
    int32_t violated;      // this step's act broke the rule
    int32_t zap_marked;    // this step's zap landed on a marked fish
    int32_t informed;      // has read the beacon this episode
    int32_t inst_type;     // public rule state
} WefTraceRowV4;

// Object-event log for offline replay (env_<id>.obj.bin, written with the trace):
// one row per pellet/waste state change. kind 0 pellet, 1 waste; event 0 present at
// reset, 1 eaten/removed, 2 spawned/regrown. Allelopathic Harvest: kind = 2 + type of the
// bush AFTER the event (2 = A, 3 = B); events 0 reset, 3 planted -> A, 4 planted -> B,
// 5 ripened, 6 eaten (the bush stays). No AH event is emitted with kind 0 or 1.
typedef struct WefObjEvent {
    int32_t episode, tick, kind, index, event;
    float x, y;
} WefObjEvent;

typedef struct Trace {
    Vec2 pos[TRACE_LENGTH];
    int index;
    int count;
} Trace;

typedef struct Client {
    int window_width;
    int window_height;
    int margin;
    bool show_field;
    bool show_sensors;
    Trace traces[MAX_AGENTS];
    // Bite overlay: frames left of the victim's flash, who bit it, per-episode counts.
    int bite_flash[MAX_AGENTS];
    int bite_from[MAX_AGENTS];
    int bites_given[MAX_AGENTS];
    int bites_taken[MAX_AGENTS];
    int bites_total;
    // WEF_VIDEO_OUT: raw frames piped to ffmpeg, one per env step, first episode only.
    FILE* video;
    int video_frames;
} Client;

typedef struct FishFood {
    Vec2 pos;
    float orientation;
    bool active;
    Vec2 intrinsic_moment;
    Vec2 induced_moment;
    // Cache for intrinsic_moment: valid while moment_set && moment_ori == orientation, so
    // the 64 cosf/sinf pairs per step are computed once per (re)spawn. Designated
    // initialisers zero both fields, which invalidates the cache.
    bool moment_set;
    float moment_ori;
    // --- Allelopathic Harvest (allelo): a permanent "bush" with a type and a ripeness flag.
    int8_t type;       // 0 = A (insulator, food_contrast_a), 1 = B (conductor, food_contrast_b)
    bool ripe;         // only ripe bushes are eaten; planting targets unripe ones
    float scale;       // induced-dipole scale for (type, ripe): conductor_scale_c(radius, contrast)
    int ripen_wait;    // steps before this unripe bush may ripen (ripen_min_steps after an eat / planting)
} FishFood;

// Inert debris (cleanup mode): no intrinsic dipole (passively invisible), induced
// dipole with its own radius/contrast so mormyromasts see it when emitting nearby.
typedef struct Waste {
    Vec2 pos;
    bool active;
    Vec2 induced_moment;
} Waste;

typedef enum FoodDistribution {
    FOOD_UNIFORM,
    FOOD_PATCHY,
    FOOD_RANDOM,
} FoodDistribution;

struct Env {
    Log log;
    Agent agents[MAX_AGENTS];
    int tag;
    int boundary_reached;
    int num_agents;
    int tick;
    int episode_length;
    unsigned int rng;
    float arena_size_x;
    float arena_size_y;
    float min_arena_size_x;
    float min_arena_size_y;
    float max_arena_size_x;
    float max_arena_size_y;
    float electric_field_radius_cm;
    float reflection_wall_range_cm;
    FoodDistribution food_distribution;
    int configured_num_food;
    float patch_radius_cm;
    float patch_radius_std_cm;
    float patch_density;
    FishAgent fish[MAX_AGENTS];
    FishFood food[MAX_FOOD];
    int num_food;
    int food_eaten;
    int eod_agent_steps;
    int collisions_fish;
    int bites;
    float food_per_fish_area;
    float episode_return;
    float amp_intrinsic_baseline[NUM_AMPULLARY];
    Client* client;
    int env_id;
    int episode;
    FILE* trace_file;
    FILE* obj_file;
    // --- Cleanup extension. Every field below is inert when cleanup == 0 and the
    // other keys sit at their wef.ini defaults (docs/wef-cleanup-v0-design.md 10.3).
    int cleanup;
    float strip_cm;
    float orchard_cm;
    int spawn_band;
    int waste_max;
    int waste_start;
    float waste_spawn_p;
    int waste_spawn_delay;
    float waste_theta;
    float waste_radius_cm;
    float waste_contrast;
    float waste_sense_range_cm;
    float regrow_p_max;
    int regrow_mode;            // 0: waste-coupled (Cleanup), 1: density-dependent at the slot's position (Harvest), 2: global logistic stock, uniform placement (Commons)
    float regrow_radius_cm;     // mode 1: neighbourhood radius for the density count
    int food_start;
    int clean_priority;
    int clean_max_items;
    int clean_cooldown_steps;
    float clean_radius_cm;
    float clean_reward;
    float clean_reward_anneal_steps;
    int bitten_freeze_steps;
    float bitten_reward;
    float bite_reward;
    float eod_cost;
    float reward_share;
    float proximity_shaping;
    int obs_extra;
    float size_min;
    float size_max;
    int desync_first_episode;
    int policy1_agents;
    int no_clean_agents;
    int first_episode_len;
    int cur_episode_length;
    int roles[MAX_AGENTS];      // scripted roles (eval only): 0 policy, 1 cleaner, 2 eater, 3 shift, 4 random
    int bot_shift_steps;
    int bot_oracle;             // 1: bots target items anywhere; 0: only within sensing range
    float bot_theta;            // role 7: eat only while food_active > bot_theta * num_food
    float regrow_allee;         // commons: no regrowth while S <= regrow_allee * K (depensation)
    int render_field;           // renderer: 1 field arrows around each fish (upstream), 0 off
    float render_field_alpha;   // renderer: arrow opacity multiplier (1 = upstream)
    float sustain_frac;         // metrics only: stock fraction at/below which eating counts as defecting
    int num_patches;            // patchy layout: patches per episode (0 = ceil(patch_density * area), upstream)
    int regrow_in_patches;      // commons (modes 2/3): regrown pellets land inside this episode's patches
    float bot_camp;             // scripted eaters: > 0 = with no pellet in range, circle the last eating spot at this radius (cm) instead of patrolling
    Vec2 patch_center[MAX_PATCHES];   // this episode's patches (patchy layout only)
    float patch_radius_ep[MAX_PATCHES];
    int num_patches_ep;         // 0 when the episode is not patchy
    int season_steps;           // regrow_mode 3: steps per season (stock grows only at season ends)
    float season_growth;        // regrow_mode 3: stock multiplier per season, capped at K (GovSim: 2)
    bool regrows;               // any regrowth mode active (fixed-length episodes, V2 traces, stock perf)
    bool collapsed;             // regrowth modes 2/3: stock fell to the collapse threshold (absorbing)
    int collapse_tick;
    int bites_by[MAX_AGENTS];
    int bites_on_eaters;
    int bites_on_defectors;
    int bites_at_risk;
    int eater_steps;
    int defector_steps;
    int frozen_steps;
    Waste waste[MAX_WASTE];
    int waste_active;
    int food_active;
    float regrow_q;             // cleanup: current regrowth multiplier (water quality)
    int cleans;
    int regrown;
    int freezes;
    int bites_in_strip;
    int open_steps;
    float waste_frac_sum;
    float raw_return_sum;
    int cleans_by[MAX_AGENTS];
    int food_by[MAX_AGENTS];
    int strip_steps_by[MAX_AGENTS];
    int clean_pending[MAX_AGENTS];
    float return_by[MAX_AGENTS];
    // --- Allelopathic Harvest (docs/wef-allelopathic-harvest-v0-design.md). Every field below
    // is inert when allelo == 0 and the keys sit at their wef.ini defaults (section 4.3).
    int allelo;                 // master gate: typed bushes, ripening, planting, taste, fixed length, V3 trace
    float ripen_lin;            // F(x) = ripen_lin x + ripen_cubic x^ripen_pow, x = n_type / num_food
    float ripen_cubic;
    float ripen_pow;
    int ripen_min_steps;        // minimum unripe time after an eat / planting
    float start_frac_a;         // initial fraction of type-A bushes (slots 0 .. round(frac K) - 1)
    float start_ripe_frac;      // bushes ripe at reset (curriculum only)
    float food_contrast_a;      // induced-dipole contrast of type A (-0.5 = today's pellet)
    float food_contrast_b;      // type B (+0.5 preset; +1.0 = the waste signature)
    float food_radius_unripe_cm;// induced-dipole radius when unripe / ripe (FOOD_RADIUS_CM = 0.25)
    float food_radius_ripe_cm;
    float unripe_intrinsic;     // multiplier on FOOD_INTRINSIC_MOMENT_C_M for unripe / ripe bushes
    float ripe_intrinsic;
    float taste_match;          // eat reward for the fish's own type
    float taste_other;          // eat reward for the other type
    float taste_other_a;        // per-group other-type reward (-1 = inherit taste_other)
    float taste_other_b;
    float taste_split;          // taste_n_a == -1: size >= taste_split -> A
    int taste_n_a;              // -1 free size draw; n: slots < n are A in band A, rest B in band B; -2 Bernoulli in-band
    float size_a_min, size_a_max;   // size bands used when taste_n_a != -1
    float size_b_min, size_b_max;
    float size_speed_exp;       // speed exponent in AH mode (SIZE_SPEED_EXPONENT = 1; 0 = taste without speed)
    int plant_mode;             // 0 planting off, 1 taste-relative bins, 2 own type only
    float plant_split;          // bin boundary on raw[3]
    int plant_bin_order;        // 0: upper bin = own type; 1: lower bin = own type
    int plant_priority;         // bite with a fish and a bush in the cone: 0 fish first, 1 bush first, 2 nearest first
    float plant_radius_cm;      // planting cone radius
    int plant_steps;            // hold after a conversion (eat_cooldown: no motion, no eating)
    int plant_cooldown_steps;   // bite cooldown after a planting bite (max with plant_steps on a conversion)
    float plant_reward;         // private shaping per conversion, after mixing (curriculum only)
    float plant_reward_anneal_steps;
    int zap_cooldown_steps;     // bite cooldown after a fish bite (BITE_COOLDOWN_STEPS = 5)
    int zap_steps;              // hold on the attacker after a zap (eat_cooldown), 0 = none
    float conv_theta;           // metrics: monoculture fraction that counts as a convention
    float bot_plant_theta;      // roles 8/9/11: plant only while n_type[target] / num_food < theta
    int bot_plant_max;          // planter bots: conversions per episode (0 = unlimited)
    float bot_plant_frac;       // planter bots: P(go planting | no ripe bush within the 5 cm sense range)
    // AH episode state
    int n_type[2];              // bushes of each type (ripe and unripe)
    int n_ripe;
    int ripened;
    int plantings;
    int plantings_by[MAX_AGENTS];
    int plantings_g[2];         // by the planter's taste
    int plant_own;              // conversions to the planter's own type (total, and by the planter's taste)
    int plant_own_g[2];
    int plant_major;
    int plant_proactive;
    int plant_noop;
    int plant_attempts;
    int eats_match;             // eats of the fish's own type
    int zaps_cross;
    int zaps_same;
    int plant_pending[MAX_AGENTS];  // conversions this step (plant_reward)
    float mono_frac_sum;
    float frac_a_sum;
    float ripe_frac_sum;
    int convention_tick;        // first tick at or above conv_theta, 0 = never
    // --- Institutions (docs/institutions-plan.md section 4). Inert when inst_mode == 0 and
    // inst_obs == 0; inst_obs needs the -DWEF_INST build (the obs row grows).
    int inst_mode;              // 0 none; 1 public beacon; 2 private signals; 3 spurious public signal
    int inst_obs;               // 1: the beacon object sits in the arena and the institution slots are filled
    int inst_fixed_type;        // AH: -1 draw the prescribed type per episode, 0 / 1 pin it
    float inst_theta;           // Commons: the public season closes while S <= theta K ...
    float inst_hyst;            // ... and reopens above (theta + hyst) K
    int inst_flip_steps;        // mode 3: AH flips the prescription every N steps; Commons toggles w.p. 1 / N per step
    int inst_mark_steps;        // violation mark duration (0 = no marks)
    float mark_zap_reward;      // bounty to the attacker for zapping a marked fish
    int mark_zap_cooldown;      // attacker bite cooldown after zapping a marked fish (-1 = BITE_COOLDOWN_STEPS)
    int mark_freeze_steps;      // freeze of a MARKED victim (-1 = bitten_freeze_steps): the sanction the institution legitimates
    int inst_pos_random;        // 0 beacon at the arena centre, 1 drawn uniformly per episode
    float inst_read_cm;         // a fish inside this radius reads the rule
    float inst_obj_radius_cm;   // the beacon's electrical signature: a waste-like conductor
    float inst_contrast;
    int inst_latch;             // 1: the obs slot keeps the last value read; 0: in range only
    // institution episode state
    int inst_type;              // public rule: AH prescribed type 0 A / 1 B; Commons 1 closed / 0 open
    int inst_type_reset;        // AH: the prescribed type drawn at reset (inst_type_a)
    Vec2 inst_pos;
    Vec2 inst_moment;           // beacon induced dipole this step
    int violations;
    int zaps_on_marked;
    int marked_steps;
    int pair_steps;             // ordered fish pairs within MORM_AGENT_RANGE_CM, summed over steps
    int pair_marked_steps;      // ... whose second fish was marked
    int informed_steps;
    float inst_agree_sum;
    int closed_steps;
    int eats_closed;
    float bounty_sum;
    int plantings_p;
    int plantings_np;
};
typedef Env Wef;

// Institutions: the rule a fish is subject to (docs/institutions-plan.md 3.2 / 3.3). AH: the
// prescribed type (public in modes 1 / 3, this fish's private draw in mode 2). Commons: whether
// the season is closed for this fish (public flag, or the stock against its private threshold).
static inline int wef_inst_rule_type(const Wef* env, int i) {
    return env->inst_mode == 2 ? env->fish[i].inst_private : env->inst_type;
}

static inline int wef_inst_closed(const Wef* env, int i) {
    if (env->inst_mode == 2) {
        return (float)env->food_active <= env->fish[i].inst_private_theta * (float)env->num_food;
    }
    return env->inst_type;
}

// The value a fish reads at the beacon: +1 / -1 (AH: type A / B; Commons: closed / open).
static inline float wef_inst_signal(const Wef* env, int i) {
    if (env->allelo) {
        return wef_inst_rule_type(env, i) == 0 ? 1.0f : -1.0f;
    }
    return wef_inst_closed(env, i) ? 1.0f : -1.0f;
}

static inline bool wef_at_beacon(const Wef* env, const FishAgent* fish) {
    float dx = fish->pos.x - env->inst_pos.x;
    float dy = fish->pos.y - env->inst_pos.y;
    return dx * dx + dy * dy <= env->inst_read_cm * env->inst_read_cm;
}

// A violation: mark the fish (public label) and count it.
static inline void wef_inst_violation(Wef* env, int i) {
    FishAgent* fish = &env->fish[i];
    env->violations++;
    fish->violations++;
    fish->trace_violated = true;
    if (env->inst_mark_steps > 0) {
        fish->mark = env->inst_mark_steps;
    }
}

// glibc's rand_r (TYPE_0 LCG, unchanged since glibc 2.x; ocean/wef/wef_test.c checks it
// against the library call): inlined so the ~400 sensor-noise draws per env-step skip the
// PLT call. Same outputs bit for bit, so the sensor streams are unchanged.
static inline int wef_rand_r(unsigned int* seed) {
    unsigned int next = *seed;
    int result;
    next *= 1103515245;
    next += 12345;
    result = (unsigned int)(next / 65536) % 2048;
    next *= 1103515245;
    next += 12345;
    result <<= 10;
    result ^= (unsigned int)(next / 65536) % 1024;
    next *= 1103515245;
    next += 12345;
    result <<= 10;
    result ^= (unsigned int)(next / 65536) % 1024;
    *seed = next;
    return result;
}

float random_uniform(Wef* env, float low, float high) {
    float unit = (float)wef_rand_r(&env->rng) / (float)RAND_MAX;
    return low + (high - low) * unit;
}

void wef_obj_event(Wef* env, int kind, int index, int event, Vec2 pos) {
    if (env->obj_file == NULL) {
        return;
    }
    WefObjEvent row = {env->episode, env->tick, kind, index, event, pos.x, pos.y};
    fwrite(&row, sizeof(row), 1, env->obj_file);
}

// Probe in cm. Sources (Mono/Dipole.p) already meters.
// Monos are always agent EODs → agent_range. Dipoles: first n_agent_dips are
// agents, the next n_food_dips food, the rest waste → agent_range / food_range /
// waste_range (paper sensor cutoffs; waste only exists in cleanup mode).
// wall_range_cm=0 → no image charges.
// Field of monopole sources only, no wall images: bit-identical to
// measure_field(env, probe_cm, mono, n_mono, NULL, 0, 0, 0, range_cm, *, 0, 0) (same
// operations in the same order) without that function's staging and range set-up. Used for
// the knollenorgan (2 poles), pellet/waste induction (2A poles) and body induction calls,
// which are ~200 of the ~450 field evaluations per env-step.
static inline Vec2 wef_mono_field(Vec2 probe_cm, const Mono* mono, int n_mono, float range_cm) {
    float pmx = probe_cm.x * CM_TO_M;
    float pmy = probe_cm.y * CM_TO_M;
    float agent_r = range_cm * CM_TO_M;
    float agent_range2 = agent_r * agent_r;
    float eps_m = FIELD_EPS_M;
    float field_x = 0.0f;
    float field_y = 0.0f;
    for (int i = 0; i < n_mono; i++) {
        float sx = mono[i].p.x;
        float sy = mono[i].p.y;
        float dx = pmx - sx;
        float dy = pmy - sy;
        if (dx * dx + dy * dy > agent_range2) {
            continue;
        }
        float dist = sqrtf(dx * dx + dy * dy) + eps_m;
        float inv_d = 1.0f / dist;
        float w = K_COULOMB * mono[i].q * inv_d * inv_d * inv_d;
        field_x += dx * w;
        field_y += dy * w;
    }
    return (Vec2){field_x, field_y};
}

Vec2 measure_field(Env* env, Vec2 probe_cm, const Mono* mono, int n_mono,
    const Dipole* dip, int n_dip, int n_agent_dips, int n_food_dips,
    float agent_range_cm, float food_range_cm, float waste_range_cm,
    float wall_range_cm) {
    float pmx = probe_cm.x * CM_TO_M;
    float pmy = probe_cm.y * CM_TO_M;
    float agent_r = agent_range_cm * CM_TO_M;
    float food_r = food_range_cm * CM_TO_M;
    float waste_r = waste_range_cm * CM_TO_M;
    float agent_range2 = agent_r * agent_r;
    float food_range2 = food_r * food_r;
    float waste_range2 = waste_r * waste_r;
    float eps_m = FIELD_EPS_M;
    float field_x = 0.0f;
    float field_y = 0.0f;

    // Stage sources only when wall images are possible (skip for induce/knollen).
    int want_walls = wall_range_cm > 0.0f;
    int near_walls[NUM_WALLS];
    int num_near_walls = 0;
    float arena_mx = 0.0f;
    float arena_my = 0.0f;
    if (want_walls) {
        arena_mx = env->arena_size_x * CM_TO_M;
        arena_my = env->arena_size_y * CM_TO_M;
        float wall_range_m = wall_range_cm * CM_TO_M;
        float wall_dist[NUM_WALLS] = {
            pmx, arena_mx - pmx, pmy, arena_my - pmy
        };
        for (int wall = 0; wall < NUM_WALLS; wall++) {
            if (wall_dist[wall] <= wall_range_m) {
                near_walls[num_near_walls++] = wall;
            }
        }
        if (num_near_walls == 0) {
            want_walls = 0;
        }
    }

    Mono mono_in[WEF_MAX_MONO];
    Dipole dip_in[WEF_MAX_DIP];
    int n_mono_in = 0;
    int n_dip_in = 0;

    for (int i = 0; i < n_mono; i++) {
        float sx = mono[i].p.x;
        float sy = mono[i].p.y;
        float dx = pmx - sx;
        float dy = pmy - sy;
        // EOD monos are always conspecific/agent sources.
        if (dx * dx + dy * dy > agent_range2) {
            continue;
        }
        if (want_walls) {
            mono_in[n_mono_in++] = mono[i];
        }
        float dist = sqrtf(dx * dx + dy * dy) + eps_m;
        float inv_d = 1.0f / dist;
        float w = K_COULOMB * mono[i].q * inv_d * inv_d * inv_d;
        field_x += dx * w;
        field_y += dy * w;
    }
    for (int i = 0; i < n_dip; i++) {
        float sx = dip[i].p.x;
        float sy = dip[i].p.y;
        float range2 = (i < n_agent_dips) ? agent_range2
            : (i < n_agent_dips + n_food_dips) ? food_range2 : waste_range2;
        float dx = pmx - sx;
        float dy = pmy - sy;
        if (dx * dx + dy * dy > range2) {
            continue;
        }
        if (want_walls) {
            dip_in[n_dip_in++] = dip[i];
        }
        float dist = sqrtf(dx * dx + dy * dy) + eps_m;
        float inv_d2 = 1.0f / (dist * dist);
        float inv_d3 = inv_d2 / dist;
        float mdot = dip[i].m.x * dx + dip[i].m.y * dy;
        float k = K_COULOMB * inv_d3;
        float t = 3.0f * mdot * inv_d2;
        field_x += k * (t * dx - dip[i].m.x);
        field_y += k * (t * dy - dip[i].m.y);
    }

    for (int w = 0; w < num_near_walls; w++) {
        int wall = near_walls[w];
        for (int k = 0; k < n_mono_in; k++) {
            float sx = mono_in[k].p.x;
            float sy = mono_in[k].p.y;
            if (wall <= WALL_RIGHT) {
                sx = wall == WALL_LEFT ? -sx : 2.0f * arena_mx - sx;
            } else {
                sy = wall == WALL_BOTTOM ? -sy : 2.0f * arena_my - sy;
            }
            float dx = pmx - sx;
            float dy = pmy - sy;
            float dist = sqrtf(dx * dx + dy * dy) + eps_m;
            float inv_d = 1.0f / dist;
            float wt = K_COULOMB * mono_in[k].q * inv_d * inv_d * inv_d;
            field_x += dx * wt;
            field_y += dy * wt;
        }
        for (int k = 0; k < n_dip_in; k++) {
            float sx = dip_in[k].p.x;
            float sy = dip_in[k].p.y;
            if (wall <= WALL_RIGHT) {
                sx = wall == WALL_LEFT ? -sx : 2.0f * arena_mx - sx;
            } else {
                sy = wall == WALL_BOTTOM ? -sy : 2.0f * arena_my - sy;
            }
            float dx = pmx - sx;
            float dy = pmy - sy;
            float dist = sqrtf(dx * dx + dy * dy) + eps_m;
            float inv_d2 = 1.0f / (dist * dist);
            float inv_d3 = inv_d2 / dist;
            float mx = dip_in[k].m.x;
            float my = dip_in[k].m.y;
            float mdot = mx * dx + my * dy;
            float kcoef = K_COULOMB * inv_d3;
            float t = 3.0f * mdot * inv_d2;
            field_x += kcoef * (t * dx - mx);
            field_y += kcoef * (t * dy - my);
        }
    }
    return (Vec2){field_x, field_y};
}
// True if any fish centre is within `range_cm` of p (cm). Used to skip field evaluations
// whose every source is provably out of range (the result would be exactly zero).
static inline int wef_any_fish_within(const Wef* env, Vec2 p, float range_cm) {
    float r2 = range_cm * range_cm;
    for (int i = 0; i < env->num_agents; i++) {
        float dx = p.x - env->fish[i].pos.x;
        float dy = p.y - env->fish[i].pos.y;
        if (dx * dx + dy * dy <= r2) {
            return 1;
        }
    }
    return 0;
}

void compute_observations(Wef* env) {
    // Build EOD poles + induced/intrinsic moments, then pack obs.
    Mono eod[WEF_MAX_MONO];
    int n_eod = 0;
    float body_scale = conductor_scale(BODY_RADIUS_CM);
    float food_scale = conductor_scale(FOOD_RADIUS_CM);
    float max_moment = EOD_CHARGE_C * BODY_RADIUS_CM;
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->fish[i];
        float c = cosf(agent->orientation);
        float s = sinf(agent->orientation);
        float q = agent->emits_eod ? EOD_CHARGE_C : 0.0f;
        agent->eod_pos[0] = (Vec2){
            c * EOD_POLE_OFFSET_CM + agent->pos.x,
            s * EOD_POLE_OFFSET_CM + agent->pos.y,
        };
        agent->eod_pos[1] = (Vec2){
            -c * EOD_POLE_OFFSET_CM + agent->pos.x,
            -s * EOD_POLE_OFFSET_CM + agent->pos.y,
        };
        agent->eod_charge[0] = q;
        agent->eod_charge[1] = -q;
        agent->intrinsic_moment = (Vec2){c * INTRINSIC_MOMENT_C_M, s * INTRINSIC_MOMENT_C_M};
        eod[n_eod++] = (Mono){to_m(agent->eod_pos[0]), q};
        eod[n_eod++] = (Mono){to_m(agent->eod_pos[1]), -q};
    }
    // EOD induces dipoles on non-emitting bodies; food gets intrinsic + induced.
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->fish[i];
        float moment_x = 0.0f;
        float moment_y = 0.0f;
        if (!agent->emits_eod) {
            // Induced by nearby EODs (conspecific EOD → body, morm agent range).
            Vec2 f = wef_mono_field(agent->pos, eod, n_eod, MORM_AGENT_RANGE_CM);
            moment_x = f.x * body_scale;
            moment_y = f.y * body_scale;
            float mag = sqrtf(moment_x * moment_x + moment_y * moment_y);
            if (mag > max_moment) {
                float s = max_moment / mag;
                moment_x *= s;
                moment_y *= s;
            }
        }
        agent->induced_moment = (Vec2){moment_x, moment_y};
    }
    for (int i = 0; i < env->num_food; i++) {
        if (!env->food[i].active) {
            env->food[i].intrinsic_moment = (Vec2){0};
            env->food[i].induced_moment = (Vec2){0};
            env->food[i].moment_set = false;
            continue;
        }
        if (!env->food[i].moment_set || env->food[i].moment_ori != env->food[i].orientation) {
            float fc = cosf(env->food[i].orientation);
            float fs = sinf(env->food[i].orientation);
            // AH: ripe bushes carry ripe_intrinsic x the pellet moment, unripe unripe_intrinsic x
            // (preset 3 / 0: the passive ripeness cue). Eat / ripen / plant reset moment_set so
            // this cache is recomputed; the default path multiplies by the unchanged constant.
            float moment = FOOD_INTRINSIC_MOMENT_C_M;
            if (env->allelo) {
                moment = FOOD_INTRINSIC_MOMENT_C_M
                    * (env->food[i].ripe ? env->ripe_intrinsic : env->unripe_intrinsic);
            }
            env->food[i].intrinsic_moment = (Vec2){
                -fs * moment,
                fc * moment,
            };
            env->food[i].moment_set = true;
            env->food[i].moment_ori = env->food[i].orientation;
        }
        // Poles sit EOD_POLE_OFFSET_CM from the fish centre: a pellet farther than
        // (range + offset + margin) from every fish sees no pole and the field is exactly 0.
        Vec2 f = {0.0f, 0.0f};
        if (wef_any_fish_within(env, env->food[i].pos, MORM_AGENT_RANGE_CM + EOD_POLE_OFFSET_CM + 0.1f)) {
            f = wef_mono_field(env->food[i].pos, eod, n_eod, MORM_AGENT_RANGE_CM);
        }
        // AH: per-bush scale (type contrast x ripeness radius); default: the folded constant.
        float scale_i = env->allelo ? env->food[i].scale : food_scale;
        env->food[i].induced_moment = (Vec2){f.x * scale_i, f.y * scale_i};
    }
    // Waste (cleanup mode): induced dipole only, from nearby EODs.
    if (env->cleanup) {
        float waste_scale = conductor_scale_c(env->waste_radius_cm, env->waste_contrast);
        // Both EOD poles of any fish whose sensors can reach the item must be in the
        // inducing set, or the reading spikes at the sensing edge (one-pole field).
        float mono_range = env->waste_sense_range_cm + BODY_RADIUS_CM + 2.0f * EOD_POLE_OFFSET_CM;
        for (int i = 0; i < env->waste_max; i++) {
            if (!env->waste[i].active) {
                env->waste[i].induced_moment = (Vec2){0};
                continue;
            }
            Vec2 f = {0.0f, 0.0f};
            if (wef_any_fish_within(env, env->waste[i].pos, mono_range + EOD_POLE_OFFSET_CM + 0.1f)) {
                f = wef_mono_field(env->waste[i].pos, eod, n_eod, mono_range);
            }
            env->waste[i].induced_moment = (Vec2){f.x * waste_scale, f.y * waste_scale};
        }
    }
    // Institution beacon (inst_obs): a static conductor sensed like a waste item (induced only,
    // waste class ranges), present in every arm of the institution build, control included.
    if (env->inst_obs) {
        float inst_scale = conductor_scale_c(env->inst_obj_radius_cm, env->inst_contrast);
        float mono_range = env->waste_sense_range_cm + BODY_RADIUS_CM + 2.0f * EOD_POLE_OFFSET_CM;
        Vec2 f = {0.0f, 0.0f};
        if (wef_any_fish_within(env, env->inst_pos, mono_range + EOD_POLE_OFFSET_CM + 0.1f)) {
            f = wef_mono_field(env->inst_pos, eod, n_eod, mono_range);
        }
        env->inst_moment = (Vec2){f.x * inst_scale, f.y * inst_scale};
    }
    // Induced + intrinsic dipoles as AoS for measure_field: [agents..., food..., waste...]
    Dipole induced[WEF_MAX_DIP];
    Dipole intrinsic[WEF_MAX_DIP];
    int n_induced = 0;
    int n_intrinsic = 0;
    for (int a = 0; a < env->num_agents; a++) {
        induced[n_induced++] = (Dipole){
            to_m(env->fish[a].pos), env->fish[a].induced_moment
        };
        intrinsic[n_intrinsic++] = (Dipole){
            to_m(env->fish[a].pos), env->fish[a].intrinsic_moment
        };
    }
    for (int f = 0; f < env->num_food; f++) {
        if (!env->food[f].active) {
            continue;
        }
        induced[n_induced++] = (Dipole){
            to_m(env->food[f].pos), env->food[f].induced_moment
        };
        intrinsic[n_intrinsic++] = (Dipole){
            to_m(env->food[f].pos), env->food[f].intrinsic_moment
        };
    }
    int n_food_staged = n_induced - env->num_agents;
    int n_waste_staged = 0;
    if (env->cleanup) {
        for (int w = 0; w < env->waste_max; w++) {
            if (!env->waste[w].active) {
                continue;
            }
            induced[n_induced++] = (Dipole){
                to_m(env->waste[w].pos), env->waste[w].induced_moment
            };
            n_waste_staged++;
        }
    }
    if (env->inst_obs) {
        // the beacon joins the waste class (same ranges and cull radius; no intrinsic moment)
        induced[n_induced++] = (Dipole){to_m(env->inst_pos), env->inst_moment};
        n_waste_staged++;
    }
    // Per-fish culling radii: every sensor sits on the BODY_RADIUS_CM ring around the fish
    // centre, so a source farther than (range + ring + margin) from the centre is out of
    // range for all 60 sensors and would be skipped by measure_field's own test. Culling once
    // per fish instead of once per sensor removes ~60x(A+F) distance tests per fish and keeps
    // the surviving sources in their original order, so the field sums are bit-identical.
    // Mormyromast ranges are the wider ones (10/5 cm vs 8/4 cm ampullary), so one cull
    // serves both organs.
    const float cull_margin_cm = BODY_RADIUS_CM + 0.1f;
    float cull_agent_m = (MORM_AGENT_RANGE_CM + cull_margin_cm) * CM_TO_M;
    float cull_food_m = (MORM_FOOD_RANGE_CM + cull_margin_cm) * CM_TO_M;
    float cull_waste_m = (env->waste_sense_range_cm + cull_margin_cm) * CM_TO_M;
    float cull_agent2 = cull_agent_m * cull_agent_m;
    float cull_food2 = cull_food_m * cull_food_m;
    float cull_waste2 = cull_waste_m * cull_waste_m;
    Dipole induced_c[WEF_MAX_DIP];
    Dipole intrinsic_c[WEF_MAX_DIP];
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->fish[i];
        obs_t* obs = env->agents[i].observations;
        int obs_idx = 0;
        float ori_c = cosf(agent->orientation);
        float ori_s = sinf(agent->orientation);
        Vec2 pm = to_m(agent->pos);
        int n_ind_c = 0;
        int n_int_c = 0;
        int n_agent_c = 0;
        int n_food_c = 0;
        for (int k = 0; k < n_induced; k++) {
            float dx = pm.x - induced[k].p.x;
            float dy = pm.y - induced[k].p.y;
            float d2 = dx * dx + dy * dy;
            int is_agent = k < env->num_agents;
            int is_food = !is_agent && k < env->num_agents + n_food_staged;
            float cull2 = is_agent ? cull_agent2 : is_food ? cull_food2 : cull_waste2;
            if (d2 > cull2) {
                continue;
            }
            induced_c[n_ind_c++] = induced[k];
            if (k < n_intrinsic) {
                // AH: an unripe bush with unripe_intrinsic 0 has an exactly-zero intrinsic
                // moment and contributes exactly 0 to every ampullary sum, so it is not staged
                // (fish always carry a nonzero moment; the default path stages everything).
                bool skip = env->allelo && is_food
                    && intrinsic[k].m.x == 0.0f && intrinsic[k].m.y == 0.0f;
                if (!skip) {
                    intrinsic_c[n_int_c++] = intrinsic[k];
                }
            }
            n_agent_c += is_agent;
            n_food_c += is_food;
        }

        bool cons_eod = false;
        for (int other = 0; other < env->num_agents; other++) {
            if (other != i && env->fish[other].emits_eod) {
                cons_eod = true;
                break;
            }
        }
        // Mormyromasts: induced field.
        for (int sensor_idx = 0; sensor_idx < NUM_MORMYROMASTS; sensor_idx++) {
            Sensor w = sensor_world_cs(&g_morm[sensor_idx], agent, ori_c, ori_s);
            Vec2 f = {0.0f, 0.0f};
            if (n_ind_c > 0) {
                f = measure_field(
                    env, w.p, NULL, 0, induced_c, n_ind_c, n_agent_c, n_food_c,
                    MORM_AGENT_RANGE_CM, MORM_FOOD_RANGE_CM, env->waste_sense_range_cm,
                    env->reflection_wall_range_cm
                );
            }
            float reading = f.x * w.n.x + f.y * w.n.y;
            if (!agent->emits_eod) {
                reading *= 100.0f;
            }
            reading *= random_uniform(env, 0.95f, 1.05f);
            obs[obs_idx++] = encode_log_sensor(
                reading, MORMYROMAST_MIN_VM, MORMYROMAST_MAX_VM
            );
        }
        // Ampullary: intrinsic.
        for (int sensor_idx = 0; sensor_idx < NUM_AMPULLARY; sensor_idx++) {
            Sensor w = sensor_world_cs(&g_amp[sensor_idx], agent, ori_c, ori_s);
            Vec2 f = {0.0f, 0.0f};
            if (n_int_c > 0) {
                f = measure_field(
                    env, w.p, NULL, 0, intrinsic_c, n_int_c, n_agent_c,
                    n_int_c - n_agent_c,
                    AMP_AGENT_RANGE_CM, AMP_FOOD_RANGE_CM, 0.0f,
                    env->reflection_wall_range_cm
                );
            }
            float noise = cons_eod ? 0.5f : 0.05f;
            float reading = (f.x * w.n.x + f.y * w.n.y -
                env->amp_intrinsic_baseline[sensor_idx]) *
                random_uniform(env, 1.0f - noise, 1.0f + noise);
            obs[obs_idx++] = encode_log_sensor(
                reading, AMPULLARY_MIN_VM, AMPULLARY_MAX_VM
            );
        }
        // Knollenorgans: conspecific EOD only.
        int metadata_start = NUM_MORMYROMASTS + NUM_AMPULLARY + NUM_KNOLLEN * (MAX_AGENTS - 1);
        int cons_slot = 0;
        bool det[MAX_AGENTS] = {false};  // institution slots: which others this fish detected
        for (int other = 0; other < MAX_AGENTS; other++) {
            if (other == i) {
                continue;
            }
            bool valid = other < env->num_agents && env->fish[other].emits_eod;
            // Both poles farther than (range + sensor ring + pole offset + margin) from
            // this fish's centre are out of range for all 12 sensors: the field is exactly
            // 0 (the sensor-noise draw is still made, so the RNG stream is unchanged).
            bool in_reach = false;
            if (valid) {
                float dx = env->fish[other].pos.x - agent->pos.x;
                float dy = env->fish[other].pos.y - agent->pos.y;
                float reach = KNOLLEN_AGENT_RANGE_CM + BODY_RADIUS_CM + EOD_POLE_OFFSET_CM + 0.1f;
                in_reach = dx * dx + dy * dy <= reach * reach;
            }
            for (int sensor_idx = 0; sensor_idx < NUM_KNOLLEN; sensor_idx++) {
                float value = 0.0f;
                if (valid && !in_reach) {
                    float raw = 0.0f * random_uniform(env, 0.95f, 1.05f);
                    if (fabsf(raw) > KNOLLEN_MIN_VM) {
                        value = raw < 0.0f ? -1.0f : 1.0f;
                    }
                } else if (valid) {
                    Sensor w = sensor_world_cs(&g_knollen[sensor_idx], agent, ori_c, ori_s);
                    Mono eod[2] = {
                        {
                            to_m(env->fish[other].eod_pos[0]),
                            env->fish[other].eod_charge[0]
                        },
                        {
                            to_m(env->fish[other].eod_pos[1]),
                            env->fish[other].eod_charge[1]
                        },
                    };
                    Vec2 f = wef_mono_field(w.p, eod, 2, KNOLLEN_AGENT_RANGE_CM);
                    float raw = (f.x * w.n.x + f.y * w.n.y) *
                        random_uniform(env, 0.95f, 1.05f);
                    if (fabsf(raw) > KNOLLEN_MIN_VM) {
                        value = raw < 0.0f ? -1.0f : 1.0f;
                    }
                }
                obs[obs_idx++] = value;
            }
            bool detected = false;
            int block_start = obs_idx - NUM_KNOLLEN;
            for (int k = 0; k < NUM_KNOLLEN; k++) {
                if (obs[block_start + k] != 0.0f) {
                    detected = true;
                    break;
                }
            }
            float metadata = -1.0f;
            det[other] = valid && detected;
            if (valid && detected) {
                metadata = agent->size - env->fish[other].size;
                metadata += random_uniform(env, -0.05f, 0.05f);
                metadata = clamp(metadata, -1.0f, 1.0f);
            }
            obs[metadata_start + cons_slot] = metadata;
            cons_slot++;
        }
        obs_idx = metadata_start + MAX_AGENTS - 1;
        for (int action_idx = 0; action_idx < ACTION_SIZE; action_idx++) {
            obs[obs_idx++] = agent->last_action[action_idx];
        }
        // Slot 103: baseline constant 0; cleanup obs_extra 1 = global waste
        // fraction ("water quality" cue), 2 = normalized x position.
        float extra = 0.0f;
        if (env->obs_extra == 1 && env->waste_max > 0) {
            extra = (float)env->waste_active / (float)env->waste_max;
        } else if (env->obs_extra == 2) {
            extra = 2.0f * agent->pos.x / env->arena_size_x - 1.0f;
        } else if (env->obs_extra == 3) {
            // commons stock S / K; AH: the ripe stock n_ripe / K
            extra = (float)(env->allelo ? env->n_ripe : env->food_active) / (float)env->num_food;
        } else if (env->obs_extra == 4) {
            extra = agent->taste == 0 ? 1.0f : -1.0f;              // AH: own taste
        } else if (env->obs_extra == 5) {
            extra = (float)(env->n_type[0] - env->n_type[1]) / (float)env->num_food;  // AH: convention cue
        } else if (env->obs_extra == 6) {
            extra = agent->last_plant_type < 0 ? -1.0f : agent->last_plant_type == 0 ? 0.0f : 1.0f;
        } else if (env->obs_extra == 7) {
            extra = (float)env->n_ripe / (float)env->num_food;      // AH: ripe fraction
        }
        obs[obs_idx++] = extra;
        obs[obs_idx++] = agent->was_bitten ? 1.0f : 0.0f;
        obs[obs_idx++] = agent->size;
        {
            int cd_max = env->clean_cooldown_steps > BITE_COOLDOWN_STEPS
                ? env->clean_cooldown_steps : BITE_COOLDOWN_STEPS;
            if (env->allelo) {
                // every bite cooldown the AH branches can set, so the slot stays <= 1
                cd_max = cd_max > env->plant_cooldown_steps ? cd_max : env->plant_cooldown_steps;
                cd_max = cd_max > env->zap_cooldown_steps ? cd_max : env->zap_cooldown_steps;
                cd_max = cd_max > env->plant_steps ? cd_max : env->plant_steps;
            }
            obs[obs_idx++] = (float)agent->bite_cooldown / (float)cd_max;
        }
        obs[obs_idx++] = clamp(
            agent->disp_ego.x / agent->max_linear_velocity, -1.0f, 1.0f);
        obs[obs_idx++] = clamp(
            agent->disp_ego.y / agent->max_linear_velocity, -1.0f, 1.0f);
        {
            // AH: a planting hold (eat_cooldown up to plant_steps) or a zap recovery (zap_steps)
            // reads 1 -> 0: every eat_cooldown the AH branches can set, so the slot stays <= 1
            int eat_max = EAT_COOLDOWN_STEPS;
            if (env->allelo) {
                eat_max = eat_max > env->plant_steps ? eat_max : env->plant_steps;
                eat_max = eat_max > env->zap_steps ? eat_max : env->zap_steps;
            }
            obs[obs_idx++] = agent->freeze > 0 ? 1.0f
                : (float)agent->eat_cooldown / (float)eat_max;
        }
#ifdef WEF_INST
        // Institution slots (docs/institutions-plan.md 4): the rule as this fish last read it at the
        // beacon (0 = never), its own mark, then one mark slot per other fish in metadata-slot
        // order, filled only for fish the knollenorgan block detected this step (a mark is a label
        // on the fish: seen when the fish is). All 0 when inst_obs == 0 (and in the control arm).
        {
            float mark_norm = env->inst_mark_steps > 0 ? 1.0f / (float)env->inst_mark_steps : 0.0f;
            obs[obs_idx++] = env->inst_obs ? agent->inst_val : 0.0f;
            obs[obs_idx++] = env->inst_obs ? (float)agent->mark * mark_norm : 0.0f;
            for (int other = 0; other < MAX_AGENTS; other++) {
                if (other == i) {
                    continue;
                }
                float m = 0.0f;
                if (env->inst_obs && det[other]) {
                    m = (float)env->fish[other].mark * mark_norm;
                }
                obs[obs_idx++] = m;
            }
        }
#else
        (void)det;
#endif
    }
}

// Allelopathic Harvest: induced-dipole scale of a bush from its type (contrast) and ripeness
// (radius). Called at reset and after every type / ripeness change.
static inline void wef_bush_scale(const Wef* env, FishFood* bush) {
    float radius = bush->ripe ? env->food_radius_ripe_cm : env->food_radius_unripe_cm;
    float contrast = bush->type == 0 ? env->food_contrast_a : env->food_contrast_b;
    bush->scale = conductor_scale_c(radius, contrast);
}

// AH: the other-type eat reward of a fish of the given taste (per-group override or taste_other).
static inline float wef_taste_other(const Wef* env, int taste) {
    float g = taste == 0 ? env->taste_other_a : env->taste_other_b;
    return g < 0.0f ? env->taste_other : g;
}

// Cleanup mode: pellet `i` uniformly in the orchard band at the far wall.
void wef_spawn_pellet(Wef* env, int i) {
    env->food[i] = (FishFood){
        .pos = {
            random_uniform(env, env->arena_size_x - env->orchard_cm, env->arena_size_x),
            random_uniform(env, 0.0f, env->arena_size_y),
        },
        .orientation = random_uniform(env, 0.0f, 2.0f * PI_F),
        .active = true,
    };
}

// Cleanup mode: waste item `i` uniformly in the strip along the x = 0 wall.
void wef_spawn_waste(Wef* env, int i) {
    env->waste[i] = (Waste){
        .pos = {
            random_uniform(env, 0.0f, env->strip_cm),
            random_uniform(env, 0.0f, env->arena_size_y),
        },
        .active = true,
    };
}

// Commons (regrow modes 2/3): where a regrown pellet lands. Uniform in the arena (upstream
// draw order: x, then y), or, with regrow_in_patches, uniform inside one of this episode's
// patches so the shared stock is concentrated where fish can find it and meet.
Vec2 wef_commons_spawn_pos(Wef* env) {
    if (env->regrow_in_patches && env->num_patches_ep > 0) {
        int p = (int)(rand_r(&env->rng) % (unsigned)env->num_patches_ep);
        float angle = random_uniform(env, 0.0f, 2.0f * PI_F);
        float r = env->patch_radius_ep[p] * sqrtf(random_uniform(env, 0.0f, 1.0f));
        return (Vec2){
            clamp(env->patch_center[p].x + r * cosf(angle), 0.0f, env->arena_size_x),
            clamp(env->patch_center[p].y + r * sinf(angle), 0.0f, env->arena_size_y),
        };
    }
    float x = random_uniform(env, 0.0f, env->arena_size_x);
    float y = random_uniform(env, 0.0f, env->arena_size_y);
    return (Vec2){x, y};
}

void puf_reset(Wef* env) {
    // Sample arena size from configured min/max
    env->arena_size_x = random_uniform(env, env->min_arena_size_x, env->max_arena_size_x);
    env->arena_size_y = random_uniform(env, env->min_arena_size_y, env->max_arena_size_y);

    // Select food distribution mode; if random, choose uniform or patchy
    FoodDistribution mode = env->food_distribution;
    if (mode == FOOD_RANDOM) {
        mode = (FoodDistribution)(rand_r(&env->rng) % 2);
    }
    env->tick = 0;
    env->food_eaten = 0;
    env->eod_agent_steps = 0;
    env->collisions_fish = 0;
    env->bites = 0;
    env->episode_return = 0.0f;
    env->waste_active = 0;
    env->food_active = 0;
    env->cleans = 0;
    env->regrown = 0;
    env->freezes = 0;
    env->bites_in_strip = 0;
    env->bites_on_eaters = 0;
    env->bites_on_defectors = 0;
    env->bites_at_risk = 0;
    env->eater_steps = 0;
    env->defector_steps = 0;
    env->frozen_steps = 0;
    env->collapsed = false;
    env->collapse_tick = 0;
    env->open_steps = 0;
    env->waste_frac_sum = 0.0f;
    env->raw_return_sum = 0.0f;
    for (int i = 0; i < MAX_AGENTS; i++) {
        env->cleans_by[i] = 0;
        env->food_by[i] = 0;
        env->strip_steps_by[i] = 0;
        env->clean_pending[i] = 0;
        env->return_by[i] = 0.0f;
        env->bites_by[i] = 0;
        env->plantings_by[i] = 0;
        env->plant_pending[i] = 0;
    }
    env->n_type[0] = 0;
    env->n_type[1] = 0;
    env->n_ripe = 0;
    env->ripened = 0;
    env->plantings = 0;
    env->plantings_g[0] = 0;
    env->plantings_g[1] = 0;
    env->plant_own = 0;
    env->plant_own_g[0] = 0;
    env->plant_own_g[1] = 0;
    env->plant_major = 0;
    env->plant_proactive = 0;
    env->plant_noop = 0;
    env->plant_attempts = 0;
    env->eats_match = 0;
    env->zaps_cross = 0;
    env->zaps_same = 0;
    env->mono_frac_sum = 0.0f;
    env->frac_a_sum = 0.0f;
    env->ripe_frac_sum = 0.0f;
    env->convention_tick = 0;
    env->violations = 0;
    env->zaps_on_marked = 0;
    env->marked_steps = 0;
    env->pair_steps = 0;
    env->pair_marked_steps = 0;
    env->informed_steps = 0;
    env->inst_agree_sum = 0.0f;
    env->closed_steps = 0;
    env->eats_closed = 0;
    env->bounty_sum = 0.0f;
    env->plantings_p = 0;
    env->plantings_np = 0;
    // Only the first episode of an env can be shortened (desync_first_episode).
    env->cur_episode_length = env->episode == 0 ? env->first_episode_len : env->episode_length;

    // Spawn fish without body overlap; place sensors in local frame
    float spawn_lo_x = 3.0f;
    float spawn_hi_x = env->arena_size_x - 3.0f;
    if (env->spawn_band) {
        spawn_lo_x = env->strip_cm + 3.0f;
        spawn_hi_x = env->arena_size_x - env->orchard_cm - 3.0f;
    }
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent agent = {0};
        agent.last_plant_type = -1;
        agent.last_eaten_slot = -1;
        agent.trace_planted_slot = -1;
        // Size draw: exactly one random_uniform per fish in every mode, so the RNG stream of
        // the default path is unchanged. AH taste (0 = A, 1 = B) follows size: taste_n_a == -1
        // thresholds the free draw at taste_split; n >= 0 pins slots < n to band A and the rest
        // to band B; -2 draws the group per fish (Bernoulli 1/2) and the size inside its band.
        if (env->allelo && env->taste_n_a == -2) {
            float u = random_uniform(env, 0.0f, 1.0f);
            agent.taste = u < 0.5f ? 1 : 0;
            float lo = agent.taste == 0 ? env->size_a_min : env->size_b_min;
            float hi = agent.taste == 0 ? env->size_a_max : env->size_b_max;
            agent.size = lo + (u - (agent.taste == 0 ? 0.5f : 0.0f)) * 2.0f * (hi - lo);
        } else if (env->allelo && env->taste_n_a >= 0) {
            agent.taste = i < env->taste_n_a ? 0 : 1;
            agent.size = agent.taste == 0
                ? random_uniform(env, env->size_a_min, env->size_a_max)
                : random_uniform(env, env->size_b_min, env->size_b_max);
        } else {
            agent.size = random_uniform(env, env->size_min, env->size_max);
            if (env->allelo) {
                agent.taste = agent.size >= env->taste_split ? 0 : 1;
            }
        }
        Vec2 pos = {0};
        for (int attempts = 0; attempts < 1000; attempts++) {
            pos = (Vec2){
                random_uniform(env, spawn_lo_x, spawn_hi_x),
                random_uniform(env, 3.0f, env->arena_size_y - 3.0f),
            };
            bool overlap = false;
            for (int j = 0; j < i; j++) {
                float dx = pos.x - env->fish[j].pos.x;
                float dy = pos.y - env->fish[j].pos.y;
                float diam = 2.0f * BODY_RADIUS_CM;
                if (dx * dx + dy * dy < diam * diam) {
                    overlap = true;
                    break;
                }
            }
            if (!overlap) {
                break;
            }
        }
        agent.pos = pos;
        agent.orientation = random_uniform(env, -PI_F, PI_F);
        // Agent's size determines its max linear/angular velocity (larger fish are faster).
        // AH: size_speed_exp (0 = taste without a speed edge); default keeps the folded constant.
        float size_mult = env->allelo ? powf(1.0f + agent.size, env->size_speed_exp)
            : powf(1.0f + agent.size, SIZE_SPEED_EXPONENT);
        agent.max_linear_velocity = (MAX_LINEAR_VELOCITY_CM_S / SIMULATION_HZ) * size_mult;
        agent.max_angular_velocity = (MAX_ANGULAR_VELOCITY_RAD_S / SIMULATION_HZ) * size_mult;
        agent.emits_eod = true;
        agent.bite_victim = -1;
        agent.can_clean = i < env->num_agents - env->no_clean_agents;
        if (env->inst_mode == 2) {
            // private signals: the same per-fish statistics as the public rule, drawn
            // independently per fish (no common knowledge). AH: a type; Commons: a threshold
            // centred on inst_theta.
            float u = random_uniform(env, 0.0f, 1.0f);
            agent.inst_private = u < 0.5f ? 0 : 1;
            agent.inst_private_theta = clamp(env->inst_theta + 0.5f * (u - 0.5f), 0.05f, 0.95f);
        }
        env->fish[i] = agent;
    }

    // Distribute food
    env->num_food = env->configured_num_food;
    if (env->cleanup) {
        // Cleanup: food_start pellets in the orchard, the rest inactive slots that
        // regrowth can fill; waste_start items in the strip.
        int n_start = env->food_start < 0 ? env->num_food : env->food_start;
        for (int i = 0; i < env->num_food; i++) {
            if (i < n_start) {
                wef_spawn_pellet(env, i);
            } else {
                env->food[i] = (FishFood){0};
            }
        }
        env->food_active = n_start;
        for (int i = 0; i < env->waste_max; i++) {
            if (i < env->waste_start) {
                wef_spawn_waste(env, i);
            } else {
                env->waste[i] = (Waste){0};
            }
        }
        env->waste_active = env->waste_start;
    } else if (mode == FOOD_UNIFORM) {
        for (int i = 0; i < env->num_food; i++) {
            env->food[i] = (FishFood){
                .pos = {
                    random_uniform(env, 0.0f, env->arena_size_x),
                    random_uniform(env, 0.0f, env->arena_size_y),
                },
                .orientation = random_uniform(env, 0.0f, 2.0f * PI_F),
                .active = true,
            };
        }
        env->food_active = env->num_food;
        env->num_patches_ep = 0;
    } else {
        // Patchy: random circular patches, food sampled uniformly in a patch disk
        float centers_x[MAX_PATCHES];
        float centers_y[MAX_PATCHES];
        float radii[MAX_PATCHES];
        float max_radius =
            fminf(env->arena_size_x, env->arena_size_y) * 0.5f;
        int num_patches = env->num_patches > 0 ? env->num_patches : (int)clamp(
            ceilf(env->patch_density *
                env->arena_size_x * env->arena_size_y),
            1, MAX_PATCHES
        );
        for (int p = 0; p < num_patches; p++) {
            centers_x[p] = random_uniform(env, 0.0f, env->arena_size_x);
            centers_y[p] = random_uniform(env, 0.0f, env->arena_size_y);
            radii[p] = clamp(
                env->patch_radius_cm + random_uniform(
                    env, -env->patch_radius_std_cm, env->patch_radius_std_cm
                ),
                1.0f, max_radius
            );
            env->patch_center[p] = (Vec2){centers_x[p], centers_y[p]};
            env->patch_radius_ep[p] = radii[p];
        }
        env->num_patches_ep = num_patches;
        for (int i = 0; i < env->num_food; i++) {
            int p = (int)(rand_r(&env->rng) % (unsigned)num_patches);
            float angle = random_uniform(env, 0.0f, 2.0f * PI_F);
            float r = radii[p] * sqrtf(random_uniform(env, 0.0f, 1.0f));
            env->food[i] = (FishFood){
                .pos = {
                    clamp(centers_x[p] + r * cosf(angle), 0.0f, env->arena_size_x),
                    clamp(centers_y[p] + r * sinf(angle), 0.0f, env->arena_size_y),
                },
                .orientation = random_uniform(env, 0.0f, 2.0f * PI_F),
                .active = true,
            };
        }
        env->food_active = env->num_food;
    }
    if (!env->cleanup && env->regrows && env->food_start >= 0 && env->food_start < env->num_food) {
        // Commons / Harvest: start below capacity (initial stock S0 = food_start); the other
        // slots stay inactive for regrowth to fill. Positions were drawn as usual above, so
        // the RNG sequence is unchanged; the reset object events below skip inactive slots.
        for (int i = env->food_start; i < env->num_food; i++) {
            env->food[i].active = false;
        }
        env->food_active = env->food_start;
    }
    if (env->allelo) {
        // Bushes: every slot stays active for the whole episode; slot i is type A for
        // i < round(start_frac_a K) (positions are random, so types are interleaved); all
        // unripe unless start_ripe_frac > 0 (curriculum: spread evenly over the slot index,
        // hence over both types, with no RNG draw).
        int n_a0 = (int)floorf(env->start_frac_a * (float)env->num_food + 0.5f);
        int n_r0 = (int)floorf(env->start_ripe_frac * (float)env->num_food + 0.5f);
        for (int i = 0; i < env->num_food; i++) {
            FishFood* bush = &env->food[i];
            bush->active = true;
            bush->type = i < n_a0 ? 0 : 1;
            bush->ripe = ((i + 1) * n_r0) / env->num_food > (i * n_r0) / env->num_food;
            bush->ripen_wait = 0;
            bush->moment_set = false;
            wef_bush_scale(env, bush);
            env->n_type[bush->type]++;
            env->n_ripe += bush->ripe;
        }
        env->food_active = env->num_food;
    }
    if (env->inst_obs || env->inst_mode > 0) {
        // Institution: beacon position and this episode's public rule (docs/institutions-plan.md 3.2).
        env->inst_pos = (Vec2){env->arena_size_x * 0.5f, env->arena_size_y * 0.5f};
        if (env->inst_pos_random) {
            env->inst_pos = (Vec2){
                random_uniform(env, 3.0f, env->arena_size_x - 3.0f),
                random_uniform(env, 3.0f, env->arena_size_y - 3.0f),
            };
        }
        env->inst_moment = (Vec2){0.0f, 0.0f};
        if (env->allelo) {
            env->inst_type = env->inst_fixed_type >= 0 ? env->inst_fixed_type
                : env->inst_mode > 0 ? (random_uniform(env, 0.0f, 1.0f) < 0.5f ? 0 : 1) : 0;
        } else {
            env->inst_type = (float)env->food_active <= env->inst_theta * (float)env->num_food ? 1 : 0;
        }
        env->inst_type_reset = env->inst_type;
    }
    // Ampullary baseline: unit intrinsic dipole at arena center
    Vec2 center = {env->arena_size_x * 0.5f, env->arena_size_y * 0.5f};
    Dipole baseline_dip[1] = {{to_m(center), (Vec2){INTRINSIC_MOMENT_C_M, 0.0f}}};
    for (int i = 0; i < NUM_AMPULLARY; i++) {
        Vec2 probe = {
            center.x + g_amp[i].p.x,
            center.y + g_amp[i].p.y,
        };
        Vec2 f = measure_field(
            env, probe, NULL, 0, baseline_dip, 1, 1, 0,
            AMP_AGENT_RANGE_CM, AMP_FOOD_RANGE_CM, 0.0f,
            env->reflection_wall_range_cm
        );
        env->amp_intrinsic_baseline[i] = f.x * g_amp[i].n.x + f.y * g_amp[i].n.y;
    }

    // Record food density per fish area
    float arena_area = env->arena_size_x * env->arena_size_y;
    env->food_per_fish_area = (float)env->num_food / (arena_area * (float)env->num_agents);

    if (env->obj_file != NULL) {
        for (int i = 0; i < env->num_food; i++) {
            if (env->food[i].active) {
                wef_obj_event(env, env->allelo ? 2 + env->food[i].type : 0, i, 0, env->food[i].pos);
            }
        }
        for (int i = 0; i < env->waste_max; i++) {
            if (env->waste[i].active) {
                wef_obj_event(env, 1, i, 0, env->waste[i].pos);
            }
        }
        fflush(env->obj_file);
    }

    compute_observations(env);
}

int wef_zone(const Wef* env, Vec2 p) {
    if (!env->cleanup) {
        return 0;
    }
    if (p.x < env->strip_cm) {
        return 1;
    }
    if (p.x > env->arena_size_x - env->orchard_cm) {
        return 2;
    }
    return 0;
}

void wef_trace_step(Wef* env) {
    WefTraceRowV4 rows[MAX_AGENTS];
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->fish[i];
        rows[i] = (WefTraceRowV4){
            .env_id = env->env_id, .episode = env->episode,
            .tick = env->tick, .agent = i,
            .x = agent->pos.x, .y = agent->pos.y,
            .orientation = agent->orientation, .size = agent->size,
            .move = agent->last_action[0], .turn = agent->last_action[1],
            .eod = agent->emits_eod, .bite = agent->bite_action,
            .bite_victim = agent->bite_victim, .was_bitten = agent->was_bitten,
            .ate = agent->ate, .collided = agent->collided,
            .reward = env->agents[i].rewards[0],
            .nearest_food = env->cleanup ? agent->trace_nearest_food
                : (agent->has_previous_food_distance ? agent->previous_food_distance : -1.0f),
            .arena_x = env->arena_size_x, .arena_y = env->arena_size_y,
            // AH: the stock is the ripe count (bushes are eaten repeatedly)
            .food_left = env->allelo ? env->n_ripe : env->num_food - env->food_eaten,
            .cleaned = agent->cleaned, .waste_left = env->waste_active,
            .food_active = env->allelo ? env->n_ripe : env->food_active, .frozen = agent->freeze,
            .zone = wef_zone(env, agent->pos),
            .taste = agent->taste, .plant_type = agent->trace_plant_type,
            .planted_slot = agent->trace_planted_slot, .ate_type = agent->trace_ate_type,
            .n_a = env->n_type[0], .n_ripe = env->n_ripe,
            .hold = agent->eat_cooldown > EAT_COOLDOWN_STEPS ? agent->eat_cooldown : 0,
            .inst_signal = agent->inst_val > 0.0f ? 1 : agent->inst_val < 0.0f ? -1 : 0,
            .mark = agent->mark, .violated = agent->trace_violated,
            .zap_marked = agent->trace_zap_marked, .informed = agent->informed,
            .inst_type = env->inst_type,
        };
    }
    if (env->inst_mode > 0 || env->inst_obs) {
        fwrite(rows, sizeof(WefTraceRowV4), env->num_agents, env->trace_file);
    } else if (env->allelo) {
        // V3: the V4 row starts with the V3 fields, so write that prefix.
        for (int i = 0; i < env->num_agents; i++) {
            fwrite(&rows[i], sizeof(WefTraceRowV3), 1, env->trace_file);
        }
    } else if (env->cleanup || env->regrows) {
        // V2: the V3 row starts with the V2 fields, so write that prefix.
        for (int i = 0; i < env->num_agents; i++) {
            fwrite(&rows[i], sizeof(WefTraceRowV2), 1, env->trace_file);
        }
    } else {
        // Baseline format: the V2 row starts with the V1 fields, so write that prefix.
        for (int i = 0; i < env->num_agents; i++) {
            fwrite(&rows[i], sizeof(WefTraceRow), 1, env->trace_file);
        }
    }
}

// Harvest regrowth (Commons Harvest, Hughes 2018): number of active pellets within
// regrow_radius_cm of slot f (excluding f itself).
int wef_food_neighbours(const Wef* env, int f) {
    float r2 = env->regrow_radius_cm * env->regrow_radius_cm;
    int k = 0;
    for (int g = 0; g < env->num_food; g++) {
        if (g == f || !env->food[g].active) {
            continue;
        }
        float dx = env->food[g].pos.x - env->food[f].pos.x;
        float dy = env->food[g].pos.y - env->food[f].pos.y;
        if (dx * dx + dy * dy <= r2) {
            k++;
        }
    }
    return k;
}

// Hughes 2018 Harvest table 0 / 0.005 / 0.02 / 0.05 for 0 / 1 / 2 / >=3 neighbours,
// normalised to regrow_p_max.
float wef_harvest_regrow_p(const Wef* env, int k) {
    static const float table[4] = {0.0f, 0.1f, 0.4f, 1.0f};
    return env->regrow_p_max * table[k > 3 ? 3 : k];
}

// Scripted roles (eval-only calibration, docs/wef-cleanup-v0-design.md 7): steer
// straight at the nearest target using env-internal positions ("oracle"), bite
// when it is inside the action cone. Writes the 4 raw action values.
// AH bots: raw bite value that decodes to the planter's own type (own = true) or the other
// type under the env's taste-relative bins (2 x plant_split = upper, 0.5 x plant_split = lower,
// swapped by plant_bin_order).
static inline float wef_bot_bin(const Wef* env, bool own) {
    bool upper = own != (env->plant_bin_order != 0);
    return upper ? 2.0f * env->plant_split : 0.5f * env->plant_split;
}

// AH bots: an active unripe bush of type != plant_type inside the plant cone (the env's target
// rule), -1 if none.
static int wef_plant_target_in_cone(const Wef* env, const FishAgent* fish, int plant_type) {
    int best = -1;
    float nearest = INFINITY;
    for (int f = 0; f < env->num_food; f++) {
        const FishFood* b = &env->food[f];
        if (!b->active || b->ripe || b->type == plant_type) {
            continue;
        }
        if (!in_forward_cone(fish, b->pos, env->plant_radius_cm, EATING_ANGLE)) {
            continue;
        }
        float dx = b->pos.x - fish->pos.x;
        float dy = b->pos.y - fish->pos.y;
        float d2 = dx * dx + dy * dy;
        if (d2 < nearest) {
            best = f;
            nearest = d2;
        }
    }
    return best;
}

void wef_bot_action(Wef* env, int i, float* raw) {
    FishAgent* fish = &env->fish[i];
    int role = env->roles[i];
    if (role == 3) {
        int phase = (env->tick + i * (env->bot_shift_steps / MAX_AGENTS)) / env->bot_shift_steps;
        role = (phase % 2 == 0) ? 1 : 2;
    }
    if (env->allelo && role == 2) {
        role = 10;  // an eater that targeted unripe bushes would be trapped: the free-rider role
    }
    // Institution roles (docs/institutions-plan.md 4; calibration only): 13 complier, 14 enforcer.
    // Both know where the beacon is and swim to it until they have read the rule. Informed, the AH
    // complier is a planter-own (role 8) when its taste is the prescribed type and a free-rider
    // (role 10) otherwise; the Commons complier is role 7 whose restraint is the season as last
    // read, and it waits AT the beacon while closed (so it keeps reading), returns to it when it
    // has nothing to eat and re-reads it every bot_shift_steps (the season changes). The enforcer hunts the nearest marked fish within the hunt range
    // whenever it can bite, and behaves as a complier otherwise.
    bool inst_role = env->inst_mode > 0 && (role == 13 || role == 14);
    bool enforcer = inst_role && role == 14;
    bool go_beacon = false;
    if (inst_role) {
        if (!fish->informed) {
            go_beacon = true;
        } else if (!env->allelo && env->tick - fish->last_read_tick > env->bot_shift_steps) {
            go_beacon = true;  // Commons: the season changes, so re-read it every bot_shift_steps
        } else if (env->allelo) {
            role = fish->taste == wef_inst_rule_type(env, i) ? 8 : 10;
        } else {
            role = 7;
        }
    }
    // Stall escape: collisions revert position but not heading, and the 3 cm avoidance
    // turn below fights the pursuit turn, so two bots next to one pellet can pin each
    // other for the rest of the episode. After 40 motionless decisions, swim off on a
    // random heading for 25 steps (role 7 holding still on purpose resets the count).
    float moved = fabsf(fish->pos.x - fish->bot_last.x) + fabsf(fish->pos.y - fish->bot_last.y);
    fish->bot_last = fish->pos;
    fish->bot_stall = (moved < 0.01f && !fish->ate) ? fish->bot_stall + 1 : 0;
    if (env->allelo && (fish->freeze > 0 || fish->eat_cooldown > EAT_COOLDOWN_STEPS)) {
        fish->bot_stall = 0;  // a planting hold or a freeze is not a stall
    }
    if (fish->bot_stall > 40 && fish->bot_escape == 0) {
        fish->bot_escape = 25;
        fish->bot_escape_turn = random_uniform(env, -1.0f, 1.0f);
    }
    if (fish->bot_escape > 0 && role != 4) {
        fish->bot_escape--;
        fish->bot_stall = 0;
        raw[0] = 2.2f;
        raw[1] = fish->bot_escape_turn;
        raw[2] = 1.0f;
        raw[3] = -1.0f;
        return;
    }
    if (role == 4) {
        raw[0] = random_uniform(env, -2.0f, 2.0f);
        raw[1] = random_uniform(env, -2.0f, 2.0f);
        raw[2] = 1.0f;
        raw[3] = random_uniform(env, -1.0f, 1.0f);
        if (env->allelo) {
            // T7 (design 7.2) assumes raw[3] ~ N(0, 1), the untrained policy's bite head: with
            // plant_split 0.6745 the bins then land 25 % upper / 25 % lower / 50 % no bite. A
            // U(-1, 1) draw would give 32.6 / 33.7 / 50. Box-Muller from two more draws; the
            // default path (allelo 0, eval only anyway) keeps its single draw.
            float u1 = random_uniform(env, 1e-7f, 1.0f);
            float u2 = random_uniform(env, 0.0f, 2.0f * PI_F);
            raw[3] = sqrtf(-2.0f * logf(u1)) * cosf(u2);
        }
        return;
    }
    // Target: nearest item of the role's kind. bot_oracle = 0 limits the search to the
    // sensing range (waste: waste_sense_range_cm, pellets: MORM_FOOD_RANGE_CM) so the
    // rates measured are those of a fish that has to find things; 1 = env positions.
    float sense = INFINITY;
    if (!env->bot_oracle) {
        sense = role == 1 ? env->waste_sense_range_cm : MORM_FOOD_RANGE_CM;
    }
    // Role 5 (Harvest cooperator): eat only pellets with >= 2 active neighbours, so
    // patches are never harvested bare. Role 6 (Harvest defector): prefer dense
    // pellets like role 5, but fall back to any pellet in range (a strict superset).
    int min_neighbours = (role == 5 || role == 6) ? 2 : 0;
    // Role 7 (Commons cooperator): a nearest-pellet eater that stops eating while the
    // global stock is at or below bot_theta * K (the cue obs_extra = 3 gives a policy).
    bool restrain = role == 7
        && (inst_role ? fish->inst_val > 0.0f
            : (float)(env->allelo ? env->n_ripe : env->food_active) <= env->bot_theta * (float)env->num_food);
    bool at_beacon = inst_role && wef_at_beacon(env, fish);
    if (inst_role && restrain && !at_beacon) {
        go_beacon = true;  // wait out the closed season at the beacon
    }
    float sense2 = sense * sense;
    Vec2 target = {0};
    float nearest = INFINITY;
    bool found = false;
    // Allelopathic Harvest roles (section 7.1). Bots read taste / type / ripe from env state.
    //   8 planter-own: eat a ripe bush within the 5 cm sense range if there is one; otherwise go
    //     to the nearest unripe bush of type != own taste (bot_oracle: anywhere) and bite it with
    //     the own-type bin; otherwise eat ripe bushes; bot_plant_theta / bot_plant_max /
    //     bot_plant_frac gate the planting.
    //   9 planter-majority: as 8 with the planted type = argmax n_type (tie -> own; bot_oracle 0
    //     counts only the bushes within 5 cm).
    //  10 free-rider: nearest ripe bush eater, never bites.
    //  11 zapper-planter: hunt the nearest non-frozen fish of the other taste and bite it; else 8.
    //  12 opportunistic planter: forage like 10; bite with the own bin only when an off-type
    //     unripe bush is already in the plant cone (no travel).
    int ah_plant_type = -1;      // type this bot wants to plant, -1 = not planting
    bool ah_plant_target = false; // the target is a bush to plant (plant cone radius, bite in cone)
    bool ah_hunt = false;         // the target is a rival fish (bite cone radius)
    bool ah_bite_now = false;     // role 12: bite this step without a travel target
    if (go_beacon) {
        target = env->inst_pos;
        found = true;
    } else if (enforcer && fish->bite_cooldown <= 0) {
        // enforcer: the nearest marked, non-frozen fish within the hunt range
        float hunt2 = env->bot_oracle ? INFINITY : MORM_AGENT_RANGE_CM * MORM_AGENT_RANGE_CM;
        for (int j = 0; j < env->num_agents; j++) {
            if (j == i || env->fish[j].freeze > 0 || env->fish[j].mark <= 0) {
                continue;
            }
            float dx = env->fish[j].pos.x - fish->pos.x;
            float dy = env->fish[j].pos.y - fish->pos.y;
            float d2 = dx * dx + dy * dy;
            if (d2 < nearest && d2 <= hunt2) {
                nearest = d2;
                target = env->fish[j].pos;
                found = true;
                ah_hunt = true;
            }
        }
    }
    if (!found && env->allelo && role >= 8) {
        int taste = fish->taste;
        // Role 11 hunts only while it can bite (bite_cooldown 0: during the zap cooldown it
        // behaves as role 8, so it plants and eats between zaps instead of trailing a rival it
        // cannot zap) and only rivals within the hunt range (bot_oracle 0: the 10 cm range at
        // which a conspecific's induced image is sensed; 1: anywhere).
        if (role == 11 && fish->bite_cooldown <= 0) {
            float hunt2 = env->bot_oracle ? INFINITY : MORM_AGENT_RANGE_CM * MORM_AGENT_RANGE_CM;
            for (int j = 0; j < env->num_agents; j++) {
                if (j == i || env->fish[j].freeze > 0 || env->fish[j].taste == taste) {
                    continue;
                }
                float dx = env->fish[j].pos.x - fish->pos.x;
                float dy = env->fish[j].pos.y - fish->pos.y;
                float d2 = dx * dx + dy * dy;
                if (d2 < nearest && d2 <= hunt2) {
                    nearest = d2;
                    target = env->fish[j].pos;
                    found = true;
                    ah_hunt = true;
                }
            }
        }
        if (!found && (role == 8 || role == 9 || role == 11 || role == 12)) {
            int want = taste;
            if (role == 9) {
                int na = env->n_type[0];
                int nb = env->n_type[1];
                if (!env->bot_oracle) {
                    na = 0;
                    nb = 0;
                    for (int f = 0; f < env->num_food; f++) {
                        float dx = env->food[f].pos.x - fish->pos.x;
                        float dy = env->food[f].pos.y - fish->pos.y;
                        if (env->food[f].active && dx * dx + dy * dy <= MORM_FOOD_RANGE_CM * MORM_FOOD_RANGE_CM) {
                            na += env->food[f].type == 0;
                            nb += env->food[f].type == 1;
                        }
                    }
                }
                want = na > nb ? 0 : nb > na ? 1 : taste;
            }
            // bot_plant_theta gates on the type the bot is about to plant (`want`): for roles 8 /
            // 11 / 12 that is its taste (7.1's n_type[taste]); for the convention follower 9 it
            // is the majority type, so the threshold is on the convention it joins.
            bool allowed = (float)env->n_type[want] < env->bot_plant_theta * (float)env->num_food
                && (env->bot_plant_max == 0 || env->plantings_by[i] < env->bot_plant_max);
            if (allowed && role == 12) {
                ah_bite_now = wef_plant_target_in_cone(env, fish, want) >= 0;
                ah_plant_type = ah_bite_now ? want : -1;
            } else if (allowed) {
                bool ripe_near = false;
                for (int f = 0; f < env->num_food && !ripe_near; f++) {
                    float dx = env->food[f].pos.x - fish->pos.x;
                    float dy = env->food[f].pos.y - fish->pos.y;
                    ripe_near = env->food[f].active && env->food[f].ripe
                        && dx * dx + dy * dy <= MORM_FOOD_RANGE_CM * MORM_FOOD_RANGE_CM;
                }
                bool go = !ripe_near
                    && (env->bot_plant_frac >= 1.0f || random_uniform(env, 0.0f, 1.0f) < env->bot_plant_frac);
                if (go) {
                    for (int f = 0; f < env->num_food; f++) {
                        const FishFood* b = &env->food[f];
                        if (!b->active || b->ripe || b->type == want) {
                            continue;
                        }
                        float dx = b->pos.x - fish->pos.x;
                        float dy = b->pos.y - fish->pos.y;
                        float d2 = dx * dx + dy * dy;
                        if (d2 < nearest && d2 <= sense2) {
                            nearest = d2;
                            target = b->pos;
                            found = true;
                            ah_plant_target = true;
                            ah_plant_type = want;
                        }
                    }
                }
            }
        }
    }
    if (found) {
        // AH: a planting or hunting target was chosen above
    } else if (role == 1) {
        for (int w = 0; w < env->waste_max; w++) {
            if (!env->waste[w].active) {
                continue;
            }
            float dx = env->waste[w].pos.x - fish->pos.x;
            float dy = env->waste[w].pos.y - fish->pos.y;
            float d2 = dx * dx + dy * dy;
            if (d2 < nearest && d2 <= sense2) {
                nearest = d2;
                target = env->waste[w].pos;
                found = true;
            }
        }
    } else {
        for (int f = 0; f < env->num_food; f++) {
            if (!env->food[f].active) {
                continue;
            }
            if (env->allelo && !env->food[f].ripe) {
                continue;  // only ripe bushes are edible
            }
            float dx = env->food[f].pos.x - fish->pos.x;
            float dy = env->food[f].pos.y - fish->pos.y;
            float d2 = dx * dx + dy * dy;
            if (d2 < nearest && d2 <= sense2 && !restrain
                    && (min_neighbours == 0 || wef_food_neighbours(env, f) >= min_neighbours)) {
                nearest = d2;
                target = env->food[f].pos;
                found = true;
            }
        }
        if (!found && role == 6) {
            for (int f = 0; f < env->num_food; f++) {
                if (!env->food[f].active) {
                    continue;
                }
                float dx = env->food[f].pos.x - fish->pos.x;
                float dy = env->food[f].pos.y - fish->pos.y;
                float d2 = dx * dx + dy * dy;
                if (d2 < nearest && d2 <= sense2) {
                    nearest = d2;
                    target = env->food[f].pos;
                    found = true;
                }
            }
        }
    }
    if (!found && inst_role) {
        // nothing to eat / plant / zap: back to the beacon (re-read the rule)
        target = env->inst_pos;
        found = true;
    }
    if (!found && role != 1 && env->bot_camp > 0.0f && fish->bot_has_eat) {
        // Camp: circle the last eating spot (a learned fish remembers where food was; patches
        // regrow in place). The orbit radius bot_camp plus the 5 cm sense range covers a patch.
        float ang = 0.03f * (float)env->tick + 1.5f * (float)i;
        target = (Vec2){
            clamp(fish->bot_last_eat.x + env->bot_camp * cosf(ang), 2.0f, env->arena_size_x - 2.0f),
            clamp(fish->bot_last_eat.y + env->bot_camp * sinf(ang), 2.0f, env->arena_size_y - 2.0f),
        };
        found = false;
    } else if (!found) {
        // Patrol the zone: sweep along y in a per-fish lane, reversing at the walls.
        // Without an orchard (Harvest mode) eaters spread their lanes across the arena.
        float lane;
        if (role == 1) {
            lane = env->strip_cm * 0.5f + ((float)i - 0.5f * (float)(env->num_agents - 1)) * 2.0f;
        } else if (env->orchard_cm > 0.0f) {
            lane = env->arena_size_x - env->orchard_cm * 0.5f
                + ((float)i - 0.5f * (float)(env->num_agents - 1)) * 2.0f;
        } else {
            lane = env->arena_size_x * ((float)i + 0.5f) / (float)env->num_agents;
        }
        if (fish->bot_dir == 0) {
            fish->bot_dir = 1;
        }
        if (fish->pos.y > env->arena_size_y - 4.0f) {
            fish->bot_dir = -1;
        } else if (fish->pos.y < 4.0f) {
            fish->bot_dir = 1;
        }
        target = (Vec2){lane, fish->bot_dir > 0 ? env->arena_size_y - 2.0f : 2.0f};
    }
    float dist = sqrtf((target.x - fish->pos.x) * (target.x - fish->pos.x)
        + (target.y - fish->pos.y) * (target.y - fish->pos.y));
    float err = wrap_angle(atan2f(target.y - fish->pos.y, target.x - fish->pos.x) - fish->orientation);
    // Collision avoidance: a fish within 3 cm and roughly ahead -> turn away from it
    // (a reverted move keeps the new heading, so pure pursuit would lock head-on).
    for (int j = 0; j < env->num_agents; j++) {
        if (j == i) {
            continue;
        }
        float dx = env->fish[j].pos.x - fish->pos.x;
        float dy = env->fish[j].pos.y - fish->pos.y;
        if (dx * dx + dy * dy > 9.0f) {
            continue;
        }
        float bearing = wrap_angle(atan2f(dy, dx) - fish->orientation);
        if (fabsf(bearing) < 1.22f) {
            err = bearing > 0.0f ? -0.5f * PI_F : 0.5f * PI_F;
            break;
        }
    }
    float turn = clamp(err / fish->max_angular_velocity, -0.99f, 0.99f);
    raw[1] = atanhf(turn);
    raw[2] = 1.0f;
    float radius = role == 1 ? env->clean_radius_cm
        : ah_plant_target ? env->plant_radius_cm
        : ah_hunt ? BITING_RADIUS_CM : EATING_RADIUS_CM;
    bool in_cone = found && in_forward_cone(fish, target, radius, EATING_ANGLE);
    // Bite/eat are decided at the pre-move pose but a clean resolves after motion:
    // stop when the item is inside the action radius, and turn in place when it is
    // close but outside the cone (pure pursuit would orbit it).
    bool aligned = fabsf(err) <= EATING_ANGLE * 0.5f;
    if (found && (dist < radius || (dist < 2.0f * radius && !aligned))) {
        raw[0] = -8.0f;                          // sigmoid -> 3e-4: no motion
    } else {
        raw[0] = fabsf(err) < 0.6f ? 2.2f : -0.85f;  // 0.9 when aligned, 0.3 while turning
    }
    // Eaters never bite (biting suppresses eating that step); cleaners bite on waste.
    raw[3] = (role == 1 && in_cone) ? 1.0f : -1.0f;
    if (ah_plant_type >= 0 && ((ah_plant_target && in_cone) || ah_bite_now)) {
        // planting bite: the bin that decodes to the wanted type; stand still so the bush is
        // still in the cone when the bite resolves after motion
        raw[3] = wef_bot_bin(env, ah_plant_type == fish->taste);
        raw[0] = -8.0f;
    } else if (ah_hunt && in_cone) {
        raw[3] = wef_bot_bin(env, true);  // zap (a stray bush in the cone would get the own type)
    }
    if (restrain && (!inst_role || at_beacon)) {
        // Eating is automatic on contact, so abstaining = holding still, not patrolling.
        raw[0] = -8.0f;
        raw[1] = 0.0f;
        fish->bot_stall = 0;
    }
}

// Cleanup: nearest active waste item inside the fish's clean cone, -1 if none.
int wef_nearest_waste_in_cone(const Wef* env, const FishAgent* fish) {
    int best = -1;
    float nearest = INFINITY;
    for (int w = 0; w < env->waste_max; w++) {
        if (!env->waste[w].active) {
            continue;
        }
        if (!in_forward_cone(fish, env->waste[w].pos, env->clean_radius_cm, EATING_ANGLE)) {
            continue;
        }
        float dx = env->waste[w].pos.x - fish->pos.x;
        float dy = env->waste[w].pos.y - fish->pos.y;
        float dist2 = dx * dx + dy * dy;
        if (dist2 < nearest) {
            best = w;
            nearest = dist2;
        }
    }
    return best;
}

// Cleanup: private cleaning shaping, linearly annealed to 0 over
// clean_reward_anneal_steps env-steps (0 = constant).
float wef_clean_reward_eff(const Wef* env) {
    if (env->clean_reward == 0.0f) {
        return 0.0f;
    }
    if (env->clean_reward_anneal_steps <= 0.0f) {
        return env->clean_reward;
    }
    float env_steps = (float)env->episode * (float)env->episode_length + (float)env->tick;
    float frac = fmaxf(0.0f, 1.0f - env_steps / env->clean_reward_anneal_steps);
    return env->clean_reward * frac;
}

// AH: private planting shaping (curriculum only), annealed like clean_reward.
float wef_plant_reward_eff(const Wef* env) {
    if (env->plant_reward == 0.0f) {
        return 0.0f;
    }
    if (env->plant_reward_anneal_steps <= 0.0f) {
        return env->plant_reward;
    }
    float env_steps = (float)env->episode * (float)env->episode_length + (float)env->tick;
    float frac = fmaxf(0.0f, 1.0f - env_steps / env->plant_reward_anneal_steps);
    return env->plant_reward * frac;
}

// Gini coefficient of non-negative counts; 0 when the total is 0.
float wef_gini(const int* x, int n) {
    float total = 0.0f;
    float abs_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        total += (float)x[i];
        for (int j = 0; j < n; j++) {
            abs_diff += fabsf((float)x[i] - (float)x[j]);
        }
    }
    if (total <= 0.0f) {
        return 0.0f;
    }
    return abs_diff / (2.0f * (float)n * total);
}

// Bite-selectivity helpers (metrics only; no effect on dynamics or RNG).
static inline int wef_recent(const Wef* env, int mark, int window) {
    return mark > 0 && env->tick - mark < window;
}

static inline int wef_is_defector(const Wef* env, const FishAgent* fish) {
    if (env->cleanup) {
        return wef_recent(env, fish->eat_mark, WEF_RECENT_STEPS)
            && !wef_recent(env, fish->clean_mark, WEF_CLEAN_MEMORY);
    }
    if (env->allelo) {
        // the paper's free rider: ate recently and has not planted within WEF_CLEAN_MEMORY
        return wef_recent(env, fish->eat_mark, WEF_RECENT_STEPS)
            && !wef_recent(env, fish->plant_mark, WEF_CLEAN_MEMORY);
    }
    if (env->regrows) {
        return wef_recent(env, fish->low_eat_mark, WEF_RECENT_STEPS);
    }
    return 0;
}

static inline int wef_at_risk(const Wef* env) {
    if (env->cleanup) {
        return env->regrow_q < 0.5f;
    }
    if (env->regrows) {
        return (float)env->food_active <= env->sustain_frac * (float)env->num_food;
    }
    return 0;
}

void puf_step(Wef* env) {
    env->tick++;
    for (int i = 0; i < env->num_agents; i++) {
        env->fish[i].was_bitten = false;
        env->fish[i].ate = false;
        env->fish[i].collided = false;
        env->fish[i].bite_victim = -1;
        env->fish[i].cleaned = 0;
        env->fish[i].trace_plant_type = 0;
        env->fish[i].trace_planted_slot = -1;
        env->fish[i].trace_ate_type = 0;
        env->fish[i].trace_plant_noop = false;
        env->fish[i].trace_violated = false;
        env->fish[i].trace_zap_marked = false;
        env->clean_pending[i] = 0;
        env->plant_pending[i] = 0;
        env->agents[i].rewards[0] = 0.0f;
        env->agents[i].terminals[0] = 0.0f;
    }

    // Actions, eat, first-order motion, collisions (per fish)
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->fish[i];
        float* raw_action = env->agents[i].actions;
        float bot_raw[ACTION_SIZE];
        if (env->roles[i] != 0) {
            wef_bot_action(env, i, bot_raw);
            raw_action = bot_raw;
        }
        float move = 1.0f / (1.0f + expf(-(float)raw_action[0]));
        float turn = tanhf((float)raw_action[1]);
        agent->emits_eod = raw_action[2] > 0.0f;
        agent->bite_action = raw_action[3] > 0.0f && agent->bite_cooldown <= 0
            && agent->freeze <= 0;
        if (agent->bite_action) {
            agent->bite_cooldown = BITE_COOLDOWN_STEPS;
        }
        agent->last_action[0] = move;
        agent->last_action[1] = turn;
        agent->last_action[2] = agent->emits_eod ? 1.0f : 0.0f;
        agent->last_action[3] = agent->bite_action ? 1.0f : 0.0f;
        if (env->allelo) {
            // Bin decode: the bite magnitude above plant_split is the upper bin. Obs slot
            // 154 reports 0 / 0.5 (lower) / 1.0 (upper) so the fish knows where it landed.
            agent->bite_upper = raw_action[3] > env->plant_split;
            if (agent->bite_action) {
                agent->last_action[3] = agent->bite_upper ? 1.0f : 0.5f;
            }
        }
        env->eod_agent_steps += agent->emits_eod ? 1 : 0;
        if (env->eod_cost != 0.0f && agent->emits_eod) {
            env->agents[i].rewards[0] += env->eod_cost;
        }

        // Effort penalty
        if (PENALIZE_EFFORT_OVER_FRAC < 1.0f) {
            float move_over = fmaxf(0.0f, fabsf(move) - PENALIZE_EFFORT_OVER_FRAC);
            float turn_over = fmaxf(0.0f, fabsf(turn) - PENALIZE_EFFORT_OVER_FRAC);
            if (move_over > 0.0f || turn_over > 0.0f) {
                env->agents[i].rewards[0] += EFFORT_OVER_REWARD * (move_over + turn_over);
            }
        }

        // Eat first active pellet in forward 45° cone within 2 cm (AH: first ripe bush; it
        // turns unripe in place and stays active, reward by taste)
        if (!agent->bite_action && agent->eat_cooldown <= 0 && agent->freeze <= 0) {
            for (int f = 0; f < env->num_food; f++) {
                if (!env->food[f].active) {
                    continue;
                }
                if (env->allelo && !env->food[f].ripe) {
                    continue;
                }
                if (!in_forward_cone(agent, env->food[f].pos, EATING_RADIUS_CM, EATING_ANGLE)) {
                    continue;
                }
                float eat_reward = EAT_REWARD;
                if (env->allelo) {
                    FishFood* bush = &env->food[f];
                    bush->ripe = false;
                    bush->ripen_wait = env->ripen_min_steps;
                    bush->moment_set = false;
                    wef_bush_scale(env, bush);
                    env->n_ripe--;
                    env->eats_match += bush->type == agent->taste;
                    agent->last_eaten_slot = f;
                    agent->last_eat_tick = env->tick;
                    agent->trace_ate_type = 1 + bush->type;
                    eat_reward = bush->type == agent->taste ? env->taste_match
                        : wef_taste_other(env, agent->taste);
                    wef_obj_event(env, 2 + bush->type, f, 6, bush->pos);
                } else {
                    env->food[f].active = false;
                    if ((float)env->food_active <= env->sustain_frac * (float)env->num_food) {
                        agent->low_eat_mark = env->tick;
                    }
                    if (env->inst_mode > 0) {
                        // the rule "do not eat while the season is closed", judged before this
                        // eat changes the stock
                        agent->rule_acts++;
                        if (wef_inst_closed(env, i)) {
                            env->eats_closed++;
                            wef_inst_violation(env, i);
                        } else {
                            agent->comply_acts++;
                        }
                    }
                    env->food_active--;
                    wef_obj_event(env, 0, f, 1, env->food[f].pos);
                }
                agent->eat_mark = env->tick;
                env->food_eaten++;
                env->food_by[i]++;
                agent->eat_cooldown = EAT_COOLDOWN_STEPS;
                agent->ate = true;
                agent->bot_last_eat = agent->pos;
                agent->bot_has_eat = true;
                env->agents[i].rewards[0] += eat_reward;
                break;
            }
        }

        Vec2 prev = agent->pos;
        float prev_ori = agent->orientation;
        float lin = 0.0f;
        float ang = 0.0f;
        if (agent->eat_cooldown <= 0 && agent->freeze <= 0) {
            lin = move * agent->max_linear_velocity;
            ang = turn * agent->max_angular_velocity;
        }
        agent->orientation = wrap_angle(agent->orientation + ang);
        agent->pos.x += cosf(agent->orientation) * lin;
        agent->pos.y += sinf(agent->orientation) * lin;

        bool collided = false;
        for (int j = 0; j < env->num_agents; j++) {
            if (j == i) {
                continue;
            }
            float dx = agent->pos.x - env->fish[j].pos.x;
            float dy = agent->pos.y - env->fish[j].pos.y;
            float diam = 2.0f * BODY_RADIUS_CM;
            if (dx * dx + dy * dy < diam * diam) {
                collided = true;
                break;
            }
        }
        if (collided) {
            agent->pos = prev;
        }
        agent->pos.x = clamp(
            agent->pos.x, BODY_RADIUS_CM,
            env->arena_size_x - BODY_RADIUS_CM
        );
        agent->pos.y = clamp(
            agent->pos.y, BODY_RADIUS_CM,
            env->arena_size_y - BODY_RADIUS_CM
        );
        float gx = agent->pos.x - prev.x;
        float gy = agent->pos.y - prev.y;
        float c = cosf(prev_ori);
        float s = sinf(prev_ori);
        // rotate ground displacement into ego frame (angle -prev_ori)
        agent->disp_ego = (Vec2){c * gx + s * gy, -s * gx + c * gy};
        agent->collided = collided;
        env->collisions_fish += collided ? 1 : 0;
        if (collided) {
            env->agents[i].rewards[0] += COLLISION_REWARD;
        }
        if (env->cleanup && agent->pos.x < env->strip_cm) {
            env->strip_steps_by[i]++;
        }
        if (env->inst_obs || env->inst_mode > 0) {
            // Reading the institution: inside inst_read_cm of the beacon. Visits are counted in
            // every arm (the control's chance-visit rate is the discovery null); the rule itself
            // is written to the slot only when there is one (inst_mode > 0).
            if (!env->inst_latch) {
                agent->inst_val = 0.0f;
            }
            if (wef_at_beacon(env, agent)) {
                agent->visits++;
                agent->last_read_tick = env->tick;
                if (!agent->informed) {
                    agent->informed = true;
                    agent->first_visit_tick = env->tick;
                }
                if (env->inst_mode > 0) {
                    agent->inst_val = wef_inst_signal(env, i);
                }
            }
        }

        // Proximity shaping
        float nearest_food = INFINITY;
        bool any_food = false;
        for (int f = 0; f < env->num_food; f++) {
            if (!env->food[f].active) {
                continue;
            }
            if (env->allelo && !env->food[f].ripe) {
                continue;  // shaping toward edible (ripe) bushes only
            }
            any_food = true;
            float dx = agent->pos.x - env->food[f].pos.x;
            float dy = agent->pos.y - env->food[f].pos.y;
            float d = sqrtf(dx * dx + dy * dy);
            if (d < nearest_food) {
                nearest_food = d;
            }
        }
        agent->trace_nearest_food = any_food ? nearest_food : -1.0f;
        if (any_food) {
            if (agent->has_previous_food_distance) {
                float arena_sum = env->arena_size_x + env->arena_size_y;
                env->agents[i].rewards[0] += env->proximity_shaping
                    * (agent->previous_food_distance - nearest_food)
                    / arena_sum;
            }
            agent->previous_food_distance = nearest_food;
            agent->has_previous_food_distance = true;
        } else if (env->cleanup || env->regrows || env->allelo) {
            // No pellet to measure against: forget the stale distance so the next
            // spawn does not pay a windfall.
            agent->has_previous_food_distance = false;
        }
    }

    // Bites (and, in cleanup mode, cleans) after all fish have moved
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* attacker = &env->fish[i];
        if (!attacker->bite_action) {
            continue;
        }
        int victim = -1;
        float nearest = INFINITY;
        for (int j = 0; j < env->num_agents; j++) {
            if (i == j) {
                continue;
            }
            if (env->fish[j].freeze > 0) {
                continue;  // immunity while frozen
            }
            if (!in_forward_cone(attacker, env->fish[j].pos, BITING_RADIUS_CM, EATING_ANGLE)) {
                continue;
            }
            float dx = env->fish[j].pos.x - attacker->pos.x;
            float dy = env->fish[j].pos.y - attacker->pos.y;
            float dist2 = dx * dx + dy * dy;
            if (dist2 < nearest) {
                victim = j;
                nearest = dist2;
            }
        }
        int waste_target = -1;
        if (env->cleanup && attacker->can_clean) {
            waste_target = wef_nearest_waste_in_cone(env, attacker);
        }
        // AH planting target (section 2.3): the nearest ACTIVE UNRIPE bush whose type differs
        // from the planted type inside the plant cone; same-type bushes never shadow it.
        int bush = -1;
        float bush_d2 = INFINITY;
        bool same_only = false;
        int plant_type = -1;
        if (env->allelo && env->plant_mode) {
            int taste = attacker->taste;
            plant_type = env->plant_mode == 2 ? taste
                : ((attacker->bite_upper ^ env->plant_bin_order) ? taste : 1 - taste);
            for (int f = 0; f < env->num_food; f++) {
                FishFood* b = &env->food[f];
                if (!b->active || b->ripe) {
                    continue;
                }
                if (!in_forward_cone(attacker, b->pos, env->plant_radius_cm, EATING_ANGLE)) {
                    continue;
                }
                if (b->type == plant_type) {
                    same_only = true;
                    continue;
                }
                float dx = b->pos.x - attacker->pos.x;
                float dy = b->pos.y - attacker->pos.y;
                float d2 = dx * dx + dy * dy;
                if (d2 < bush_d2) {
                    bush = f;
                    bush_d2 = d2;
                }
            }
        }
        // plant_priority: 0 fish first (today's zap semantics), 1 bush first, 2 nearest first
        bool zap = victim >= 0;
        if (zap && bush >= 0) {
            zap = env->plant_priority == 0 || (env->plant_priority == 2 && nearest <= bush_d2);
        }
        if (waste_target >= 0 && (env->clean_priority == 1 || victim < 0)) {
            // CLEAN: remove up to clean_max_items nearest items; no reward here
            // (clean_reward is applied privately after mixing).
            int removed = 0;
            while (waste_target >= 0 && removed < env->clean_max_items) {
                env->waste[waste_target].active = false;
                env->waste_active--;
                wef_obj_event(env, 1, waste_target, 1, env->waste[waste_target].pos);
                removed++;
                waste_target = removed < env->clean_max_items
                    ? wef_nearest_waste_in_cone(env, attacker) : -1;
            }
            env->cleans += removed;
            env->cleans_by[i] += removed;
            attacker->clean_mark = env->tick;
            attacker->cleaned = removed;
            env->clean_pending[i] = removed;
            attacker->bite_cooldown = env->clean_cooldown_steps;
        } else if (zap) {
            env->fish[victim].was_bitten = true;
            attacker->bite_victim = victim;
            env->bites++;
            env->bites_by[i]++;
            env->bites_on_eaters += wef_recent(env, env->fish[victim].eat_mark, WEF_RECENT_STEPS);
            env->bites_on_defectors += wef_is_defector(env, &env->fish[victim]);
            env->bites_at_risk += wef_at_risk(env);
            // Reference: is_bitten * (1 + size_diff), size_diff ∈ [-1, 1] → factor ∈ [0, 2]
            float size_difference = attacker->size - env->fish[victim].size;
            env->agents[victim].rewards[0] += env->bitten_reward * (1.0f + size_difference);
            env->agents[i].rewards[0] += env->bite_reward;
            bool marked_victim = env->inst_mode > 0 && env->fish[victim].mark > 0;
            // institutions: a marked victim's freeze is mark_freeze_steps when set (the sanction the
            // rule legitimates); unmarked victims keep bitten_freeze_steps
            int freeze_steps = marked_victim && env->mark_freeze_steps >= 0
                ? env->mark_freeze_steps : env->bitten_freeze_steps;
            if (freeze_steps > 0) {
                // AH: the freeze stacks on any planting hold the victim is serving (section
                // 2.5, Q14), so a zap early in a hold still costs the planter the full sanction.
                // A hold is an eat_cooldown above the ordinary EAT_COOLDOWN_STEPS (the trace's
                // `hold` field and the renderer use the same test); a victim serving the plain
                // 3-step eat cooldown is frozen for bitten_freeze_steps exactly.
                int hold = env->allelo && env->fish[victim].eat_cooldown > EAT_COOLDOWN_STEPS
                    ? env->fish[victim].eat_cooldown : 0;
                env->fish[victim].freeze = freeze_steps + hold;
                env->freezes++;
            }
            if (env->cleanup && attacker->pos.x < env->strip_cm) {
                env->bites_in_strip++;
            }
            if (marked_victim) {
                // sanctioning a labelled rule-breaker: cheaper (mark_zap_cooldown) and, with a
                // bounty configured, paid (docs/institutions-plan.md 3.2 item 4; v0 bounty 0)
                env->zaps_on_marked++;
                attacker->trace_zap_marked = true;
                env->agents[i].rewards[0] += env->mark_zap_reward;
                env->bounty_sum += env->mark_zap_reward;
            }
            if (env->allelo || env->inst_mode > 0) {
                attacker->bite_cooldown = marked_victim
                    ? (env->mark_zap_cooldown >= 0 ? env->mark_zap_cooldown : BITE_COOLDOWN_STEPS)
                    : env->zap_cooldown_steps;
            }
            if (env->allelo) {
                if (env->zap_steps > attacker->eat_cooldown) {
                    attacker->eat_cooldown = env->zap_steps;  // head-butt recovery (pricing lever)
                }
                if (env->fish[victim].taste != attacker->taste) {
                    env->zaps_cross++;
                } else {
                    env->zaps_same++;
                }
            }
        } else if (bush >= 0) {
            // PLANT (conversion): recolour the bush, hold the planter for plant_steps through
            // eat_cooldown (no motion, no eating), bite cooldown max(plant_cooldown, hold).
            FishFood* b = &env->food[bush];
            int old_type = b->type;
            int major = env->n_type[0] > env->n_type[1] ? 0 : env->n_type[1] > env->n_type[0] ? 1 : -1;
            b->type = (int8_t)plant_type;
            b->ripen_wait = env->ripen_min_steps;
            b->moment_set = false;
            wef_bush_scale(env, b);
            env->n_type[old_type]--;
            env->n_type[plant_type]++;
            env->plantings++;
            env->plantings_by[i]++;
            env->plantings_g[attacker->taste]++;
            env->plant_own += plant_type == attacker->taste;
            env->plant_own_g[attacker->taste] += plant_type == attacker->taste;
            env->plant_major += plant_type == major;
            env->plant_proactive += !(bush == attacker->last_eaten_slot
                && env->tick - attacker->last_eat_tick <= 3);
            env->plant_pending[i]++;
            if (env->inst_mode > 0) {
                // the rule "plant the prescribed type" (plant_mode 2: the prescribed group
                // complies whenever it plants, the other group violates whenever it plants)
                int rule = wef_inst_rule_type(env, i);
                attacker->rule_acts++;
                if (attacker->taste == rule) {
                    env->plantings_p++;
                } else {
                    env->plantings_np++;
                }
                if (plant_type == rule) {
                    attacker->comply_acts++;
                } else {
                    wef_inst_violation(env, i);
                }
            }
            attacker->last_plant_type = plant_type;
            attacker->plant_mark = env->tick;
            attacker->trace_plant_type = 1 + plant_type;
            attacker->trace_planted_slot = bush;
            wef_obj_event(env, 2 + plant_type, bush, 3 + plant_type, b->pos);
            if (env->plant_steps > attacker->eat_cooldown) {
                attacker->eat_cooldown = env->plant_steps;  // THE HOLD
            }
            attacker->bite_cooldown = env->plant_cooldown_steps > env->plant_steps
                ? env->plant_cooldown_steps : env->plant_steps;
        } else if (same_only) {
            // same-type unripe bushes only: a no-op that spends the planting cooldown, no hold
            env->plant_noop++;
            attacker->trace_plant_noop = true;
            attacker->bite_cooldown = env->plant_cooldown_steps;
        }
        if (env->allelo && !zap) {
            // every AH bite that did not zap a fish (the paper's "total planting"): under
            // plant_priority 1 / 2 a conversion made with a fish in the cone counts too, so
            // plantings + plant_noop <= plant_attempts for every priority (design 2.3 wrote
            // `victim < 0`, identical under the preset's plant_priority 0).
            env->plant_attempts++;
        }
    }

    // Cleanup dynamics: waste inflow (DirtSpawner) and pellet regrowth (AppleGrow)
    if (env->cleanup) {
        if (env->waste_spawn_p > 0.0f && env->tick > env->waste_spawn_delay
                && env->waste_active < env->waste_max
                && random_uniform(env, 0.0f, 1.0f) < env->waste_spawn_p) {
            for (int w = 0; w < env->waste_max; w++) {
                if (!env->waste[w].active) {
                    wef_spawn_waste(env, w);
                    env->waste_active++;
                    wef_obj_event(env, 1, w, 2, env->waste[w].pos);
                    break;
                }
            }
        }
        float q = 1.0f;
        if (env->waste_max > 0) {
            q = clamp(1.0f - (float)env->waste_active
                / (env->waste_theta * (float)env->waste_max), 0.0f, 1.0f);
        }
        env->open_steps += q > 0.0f ? 1 : 0;
        if (env->waste_max > 0) {
            env->waste_frac_sum += (float)env->waste_active / (float)env->waste_max;
        }
        env->regrow_q = q;
    }
    if (env->regrows) {
        bool any_spawned = false;
        if (env->regrow_mode == 3) {
            // Seasonal (GovSim-style): no growth within a season; at each season end the
            // stock is multiplied by season_growth (capped at K) unless it is at or below
            // the collapse threshold regrow_allee * K, in which case it never regrows.
            if (!env->collapsed && env->tick % env->season_steps == 0) {
                int stock = env->food_active;
                if ((float)stock <= env->regrow_allee * (float)env->num_food || stock == 0) {
                    env->collapsed = true;
                    env->collapse_tick = env->tick;
                } else {
                    int target = (int)floorf((float)stock * env->season_growth + 0.5f);
                    target = target > env->num_food ? env->num_food : target;
                    for (int f = 0; f < env->num_food && env->food_active < target; f++) {
                        if (env->food[f].active) {
                            continue;
                        }
                        env->food[f] = (FishFood){
                            .pos = wef_commons_spawn_pos(env),
                            .orientation = random_uniform(env, 0.0f, 2.0f * PI_F),
                            .active = true,
                        };
                        env->food_active++;
                        env->regrown++;
                        any_spawned = true;
                        wef_obj_event(env, 0, f, 2, env->food[f].pos);
                    }
                }
            }
        } else if (env->regrow_mode == 1) {
            // Harvest: density-dependent regrowth at the slot's own position. Counts
            // use the pre-regrowth state so slot order does not matter.
            float p_slot[MAX_FOOD];
            for (int f = 0; f < env->num_food; f++) {
                p_slot[f] = env->food[f].active ? 0.0f
                    : wef_harvest_regrow_p(env, wef_food_neighbours(env, f));
            }
            for (int f = 0; f < env->num_food; f++) {
                if (p_slot[f] > 0.0f && random_uniform(env, 0.0f, 1.0f) < p_slot[f]) {
                    env->food[f].active = true;
                    env->food[f].orientation = random_uniform(env, 0.0f, 2.0f * PI_F);
                    env->food_active++;
                    env->regrown++;
                    any_spawned = true;
                    wef_obj_event(env, 0, f, 2, env->food[f].pos);
                }
            }
        } else if (env->regrow_mode == 2) {
            // Commons: one global stock S with logistic growth. Each empty slot regrows
            // w.p. p_max * S / K, so expected growth is p_max * S * (K - S) / K (zero at
            // S = 0, max at K / 2); new pellets land uniformly in the arena, so what one
            // fish leaves uneaten grows the stock for everyone. Depensation: no growth
            // at or below the critical stock A = regrow_allee * K, so overharvesting
            // past A collapses the commons for the rest of the episode.
            float stock = (float)env->food_active;
            float p = stock > env->regrow_allee * (float)env->num_food
                ? env->regrow_p_max * stock / (float)env->num_food : 0.0f;
            for (int f = 0; f < env->num_food; f++) {
                if (env->food[f].active || random_uniform(env, 0.0f, 1.0f) >= p) {
                    continue;
                }
                env->food[f] = (FishFood){
                    .pos = wef_commons_spawn_pos(env),
                    .orientation = random_uniform(env, 0.0f, 2.0f * PI_F),
                    .active = true,
                };
                env->food_active++;
                env->regrown++;
                any_spawned = true;
                wef_obj_event(env, 0, f, 2, env->food[f].pos);
            }
        } else if (env->cleanup && env->regrow_q > 0.0f) {
            // Cleanup: uniform in the orchard, rate set by water quality.
            float p = env->regrow_p_max * env->regrow_q;
            for (int f = 0; f < env->num_food; f++) {
                if (env->food[f].active) {
                    continue;
                }
                if (random_uniform(env, 0.0f, 1.0f) < p) {
                    wef_spawn_pellet(env, f);
                    env->food_active++;
                    env->regrown++;
                    any_spawned = true;
                    wef_obj_event(env, 0, f, 2, env->food[f].pos);
                }
            }
        }
        if (any_spawned) {
            for (int i = 0; i < env->num_agents; i++) {
                env->fish[i].has_previous_food_distance = false;
            }
        }
        if (env->regrow_mode == 2 && !env->collapsed
                && ((float)env->food_active <= env->regrow_allee * (float)env->num_food
                    || env->food_active == 0)) {
            env->collapsed = true;  // absorbing: growth is zero at or below the threshold
            env->collapse_tick = env->tick;
        }
    }
    if (env->allelo) {
        // Ripening (Bernoulli, memoryless): each unripe bush of type k with ripen_wait == 0
        // ripens w.p. F(n_k / K) = ripen_lin x + ripen_cubic x^ripen_pow, one draw per bush.
        float p_type[2];
        for (int k = 0; k < 2; k++) {
            float x = (float)env->n_type[k] / (float)env->num_food;
            p_type[k] = env->ripen_lin * x + env->ripen_cubic * powf(x, env->ripen_pow);
        }
        for (int f = 0; f < env->num_food; f++) {
            FishFood* b = &env->food[f];
            if (!b->active || b->ripe) {
                continue;
            }
            if (b->ripen_wait > 0) {
                b->ripen_wait--;
                continue;
            }
            if (random_uniform(env, 0.0f, 1.0f) < p_type[b->type]) {
                b->ripe = true;
                b->moment_set = false;
                wef_bush_scale(env, b);
                env->n_ripe++;
                env->ripened++;
                wef_obj_event(env, 2 + b->type, f, 5, b->pos);
                // Proximity shaping: a fish whose nearest ripe bush is now this one would be paid
                // a windfall (previous distance - new distance) for standing still, so forget its
                // tracked distance. Only the fish it is nearer for: at the AH-strict flux (~0.3
                // ripenings per step) resetting everyone would switch shaping off almost entirely.
                if (env->proximity_shaping != 0.0f) {
                    for (int i = 0; i < env->num_agents; i++) {
                        FishAgent* a = &env->fish[i];
                        if (!a->has_previous_food_distance) {
                            continue;
                        }
                        float dx = a->pos.x - b->pos.x;
                        float dy = a->pos.y - b->pos.y;
                        if (sqrtf(dx * dx + dy * dy) < a->previous_food_distance) {
                            a->has_previous_food_distance = false;
                        }
                    }
                }
            }
        }
        // Per-step accumulators for the composition metrics (section 5.1).
        int n_max = env->n_type[0] > env->n_type[1] ? env->n_type[0] : env->n_type[1];
        env->mono_frac_sum += (float)n_max / (float)env->num_food;
        env->frac_a_sum += (float)env->n_type[0] / (float)env->num_food;
        env->ripe_frac_sum += (float)env->n_ripe / (float)env->num_food;
        if (env->convention_tick == 0 && (float)n_max >= env->conv_theta * (float)env->num_food) {
            env->convention_tick = env->tick;
        }
    }

    if (env->inst_mode > 0 || env->inst_obs) {
        if (env->inst_mode > 0 && !env->allelo) {
            // Commons: the public season from the stock (hysteresis), or, in mode 3, a random
            // on / off schedule with mean block length inst_flip_steps (the spurious rule)
            if (env->inst_mode == 3) {
                if (random_uniform(env, 0.0f, 1.0f) < 1.0f / (float)env->inst_flip_steps) {
                    env->inst_type = 1 - env->inst_type;
                }
            } else {
                float stock = (float)env->food_active;
                float k = (float)env->num_food;
                if (env->inst_type == 0 && stock <= env->inst_theta * k) {
                    env->inst_type = 1;
                } else if (env->inst_type == 1 && stock > (env->inst_theta + env->inst_hyst) * k) {
                    env->inst_type = 0;
                }
            }
            env->closed_steps += env->inst_type;
        } else if (env->inst_mode == 3 && env->allelo && env->inst_flip_steps > 0
                && env->tick % env->inst_flip_steps == 0) {
            env->inst_type = 1 - env->inst_type;  // AH spurious rule: the prescription flips
        }
        if (env->allelo) {
            env->inst_agree_sum += (float)env->n_type[env->inst_type] / (float)env->num_food;
        }
        // marks, exposure (the mark-blind null for zap targeting) and informed fish
        for (int i = 0; i < env->num_agents; i++) {
            env->marked_steps += env->fish[i].mark > 0;
            env->informed_steps += env->fish[i].informed;
            for (int j = 0; j < env->num_agents; j++) {
                if (j == i) {
                    continue;
                }
                float dx = env->fish[j].pos.x - env->fish[i].pos.x;
                float dy = env->fish[j].pos.y - env->fish[i].pos.y;
                if (dx * dx + dy * dy <= MORM_AGENT_RANGE_CM * MORM_AGENT_RANGE_CM) {
                    env->pair_steps++;
                    env->pair_marked_steps += env->fish[j].mark > 0;
                }
            }
        }
    }
    for (int i = 0; i < env->num_agents; i++) {
        env->eater_steps += wef_recent(env, env->fish[i].eat_mark, WEF_RECENT_STEPS);
        env->defector_steps += wef_is_defector(env, &env->fish[i]);
        env->frozen_steps += env->fish[i].freeze > 0;
    }
    for (int i = 0; i < env->num_agents; i++) {
        env->fish[i].eat_cooldown -= env->fish[i].eat_cooldown > 0;
        env->fish[i].bite_cooldown -= env->fish[i].bite_cooldown > 0;
        env->fish[i].freeze -= env->fish[i].freeze > 0;
        env->fish[i].mark -= env->fish[i].mark > 0;
        // Raw (pre-mixing, pre-shaping) per-fish return: what the metrics report.
        env->return_by[i] += env->agents[i].rewards[0];
        env->raw_return_sum += env->agents[i].rewards[0];
    }

    if (env->trace_file != NULL) {
        wef_trace_step(env);
    }

    // Reward mixing (common / exchanged reward) and private cleaning shaping.
    if (env->reward_share > 0.0f) {
        float mean = 0.0f;
        for (int i = 0; i < env->num_agents; i++) {
            mean += env->agents[i].rewards[0];
        }
        mean /= (float)env->num_agents;
        float w = env->reward_share;
        for (int i = 0; i < env->num_agents; i++) {
            env->agents[i].rewards[0] = (1.0f - w) * env->agents[i].rewards[0] + w * mean;
        }
    }
    if (env->clean_reward != 0.0f) {
        float eff = wef_clean_reward_eff(env);
        for (int i = 0; i < env->num_agents; i++) {
            if (env->clean_pending[i] > 0) {
                env->agents[i].rewards[0] += (float)env->clean_pending[i] * eff;
            }
        }
    }
    if (env->allelo && env->plant_reward != 0.0f) {
        float eff = wef_plant_reward_eff(env);
        for (int i = 0; i < env->num_agents; i++) {
            if (env->plant_pending[i] > 0) {
                env->agents[i].rewards[0] += (float)env->plant_pending[i] * eff;
            }
        }
    }
    for (int i = 0; i < env->num_agents; i++) {
        env->episode_return += env->agents[i].rewards[0];
    }

    compute_observations(env);

    if (env->tick >= env->cur_episode_length
            || (!env->cleanup && !env->regrows && !env->allelo && env->food_eaten == env->num_food)) {
        env->episode++;
        if (env->trace_file != NULL) {
            fflush(env->trace_file);
        }
        if (env->obj_file != NULL) {
            fflush(env->obj_file);
        }
        float ticks = (float)env->tick;
        env->log.episode_length += ticks;
        env->log.episode_return += env->episode_return;
        env->log.score += env->raw_return_sum;
        float waste_frac = env->waste_max > 0 ? env->waste_frac_sum / ticks : 0.0f;
        env->log.perf += env->allelo ? env->mono_frac_sum / ticks  // AH: the paper's m-bar
            : env->cleanup ? 1.0f - waste_frac
            : env->regrows ? (float)env->food_active / (float)env->num_food  // Harvest/Commons: stock left
            : (float)env->food_eaten / (float)env->num_food;
        env->log.food_eaten_mean +=(float)env->food_eaten / (float)env->num_agents;
        env->log.eod_rate += (float)env->eod_agent_steps / (float)(env->tick * env->num_agents);
        env->log.collisions_fish += (float)env->collisions_fish;
        env->log.bites += (float)env->bites;
        env->log.food_per_fish_area += env->food_per_fish_area;
        // Cleanup metrics (all zero-valued in baseline except equality)
        env->log.collective_food += (float)env->food_eaten;
        env->log.waste_frac += waste_frac;
        env->log.frac_open += env->cleanup ? (float)env->open_steps / ticks : 1.0f;
        env->log.cleans += (float)env->cleans;
        env->log.regrown += (float)env->regrown;
        env->log.equality += 1.0f - wef_gini(env->food_by, env->num_agents);
        env->log.clean_gini += wef_gini(env->cleans_by, env->num_agents);
        int max_cleans = 0;
        int strip_steps = 0;
        for (int i = 0; i < env->num_agents; i++) {
            max_cleans = env->cleans_by[i] > max_cleans ? env->cleans_by[i] : max_cleans;
            strip_steps += env->strip_steps_by[i];
        }
        env->log.clean_max_share += env->cleans > 0 ? (float)max_cleans / (float)env->cleans : 0.0f;
        env->log.strip_frac += (float)strip_steps / (ticks * (float)env->num_agents);
        env->log.bites_in_strip += (float)env->bites_in_strip;
        env->log.freezes += (float)env->freezes;
        {
            float fish_steps = ticks * (float)env->num_agents;
            int top = 0;
            for (int i = 0; i < env->num_agents; i++) {
                top = env->bites_by[i] > top ? env->bites_by[i] : top;
            }
            env->log.bites_on_eaters += (float)env->bites_on_eaters;
            env->log.bites_on_defectors += (float)env->bites_on_defectors;
            env->log.bites_at_risk += (float)env->bites_at_risk;
            env->log.bites_top += (float)top;
            env->log.eater_frac += (float)env->eater_steps / fish_steps;
            env->log.defector_frac += (float)env->defector_steps / fish_steps;
            env->log.frozen_frac += (float)env->frozen_steps / fish_steps;
            env->log.collapsed += env->collapsed ? 1.0f : 0.0f;
            env->log.survival_frac += env->collapsed ? (float)env->collapse_tick / ticks : 1.0f;
        }
        float pol_sum[2] = {0.0f, 0.0f};
        int pol_n[2] = {0, 0};
        for (int i = 0; i < env->num_agents; i++) {
            int p = env->agents[i].policy == 1 ? 1 : 0;
            pol_sum[p] += env->return_by[i];
            pol_n[p]++;
        }
        env->log.policy_0_score += pol_n[0] > 0 ? pol_sum[0] / (float)pol_n[0] : 0.0f;
        env->log.policy_1_score += pol_n[1] > 0 ? pol_sum[1] / (float)pol_n[1] : 0.0f;
        if (env->allelo) {
            float k = (float)env->num_food;
            float frac_a_final = (float)env->n_type[0] / k;
            float frac_a_mean = env->frac_a_sum / ticks;
            float taste_sum[2] = {0.0f, 0.0f};
            int taste_n[2] = {0, 0};
            for (int i = 0; i < env->num_agents; i++) {
                int t = env->fish[i].taste == 0 ? 0 : 1;
                taste_sum[t] += env->return_by[i];
                taste_n[t]++;
            }
            float g_a = (float)taste_n[0] / (float)env->num_agents;
            int major_taste = taste_n[0] > taste_n[1] ? 0 : taste_n[1] > taste_n[0] ? 1 : -1;
            env->log.mono_frac += env->mono_frac_sum / ticks;
            env->log.mono_final += frac_a_final > 1.0f - frac_a_final ? frac_a_final : 1.0f - frac_a_final;
            env->log.frac_a_final += frac_a_final;
            env->log.frac_a_mean += frac_a_mean;
            env->log.majority_frac_final += major_taste < 0 ? 0.5f
                : major_taste == 0 ? frac_a_final : 1.0f - frac_a_final;
            env->log.conv_c += fabsf(frac_a_mean - g_a);
            env->log.time_to_convention += env->convention_tick > 0
                ? (float)env->convention_tick / ticks : 1.0f;
            env->log.ripened += (float)env->ripened;
            env->log.ripe_frac += env->ripe_frac_sum / ticks;
            env->log.plantings += (float)env->plantings;
            env->log.plantings_a += (float)env->plantings_g[0];
            env->log.plantings_b += (float)env->plantings_g[1];
            env->log.plant_attempts += (float)env->plant_attempts;
            env->log.plant_own_frac += env->plantings > 0 ? (float)env->plant_own / (float)env->plantings : 0.0f;
            env->log.plant_own_frac_a += env->plantings_g[0] > 0
                ? (float)env->plant_own_g[0] / (float)env->plantings_g[0] : 0.0f;
            env->log.plant_own_frac_b += env->plantings_g[1] > 0
                ? (float)env->plant_own_g[1] / (float)env->plantings_g[1] : 0.0f;
            env->log.plant_major_frac += env->plantings > 0 ? (float)env->plant_major / (float)env->plantings : 0.0f;
            env->log.plant_proactive += (float)env->plant_proactive;
            env->log.plant_noop += (float)env->plant_noop;
            env->log.plant_gini += wef_gini(env->plantings_by, env->num_agents);
            env->log.eaten_match_frac += env->food_eaten > 0 ? (float)env->eats_match / (float)env->food_eaten : 0.0f;
            env->log.zaps_cross += (float)env->zaps_cross;
            env->log.zaps_same += (float)env->zaps_same;
            env->log.taste_a_return += taste_n[0] > 0 ? taste_sum[0] / (float)taste_n[0] : 0.0f;
            env->log.taste_b_return += taste_n[1] > 0 ? taste_sum[1] / (float)taste_n[1] : 0.0f;
            env->log.taste_a_n += (float)taste_n[0];
        }
        if (env->inst_mode > 0 || env->inst_obs) {
            float fish_steps = ticks * (float)env->num_agents;
            int comply = 0;
            int acts = 0;
            int comply_inf = 0;
            int acts_inf = 0;
            int informed_n = 0;
            int visits = 0;
            float first_visit = 0.0f;
            for (int i = 0; i < env->num_agents; i++) {
                FishAgent* f = &env->fish[i];
                comply += f->comply_acts;
                acts += f->rule_acts;
                visits += f->visits;
                if (f->informed) {
                    informed_n++;
                    comply_inf += f->comply_acts;
                    acts_inf += f->rule_acts;
                    first_visit += (float)f->first_visit_tick / ticks;
                } else {
                    first_visit += 1.0f;
                }
            }
            float k = (float)env->num_food;
            env->log.inst_type_a += env->allelo && env->inst_type_reset == 0 ? 1.0f : 0.0f;
            env->log.comply_frac += acts > 0 ? (float)comply / (float)acts : 1.0f;
            env->log.comply_informed += acts_inf > 0 ? (float)comply_inf / (float)acts_inf : 1.0f;
            env->log.violations += (float)env->violations;
            env->log.marked_frac += (float)env->marked_steps / fish_steps;
            env->log.marked_exposure += env->pair_steps > 0
                ? (float)env->pair_marked_steps / (float)env->pair_steps : 0.0f;
            env->log.zaps_on_marked += (float)env->zaps_on_marked;
            env->log.zaps_on_marked_share += env->bites > 0 ? (float)env->zaps_on_marked / (float)env->bites : 0.0f;
            env->log.inst_agree_final += env->allelo ? (float)env->n_type[env->inst_type] / k : 0.0f;
            env->log.inst_agree_mean += env->allelo ? env->inst_agree_sum / ticks : 0.0f;
            env->log.plantings_p += (float)env->plantings_p;
            env->log.plantings_np += (float)env->plantings_np;
            env->log.closed_frac += env->allelo ? 0.0f : (float)env->closed_steps / ticks;
            env->log.eats_closed += (float)env->eats_closed;
            env->log.mark_zap_bounty += env->bounty_sum;
            env->log.beacon_first_visit += first_visit / (float)env->num_agents;
            env->log.informed_frac += (float)informed_n / (float)env->num_agents;
            env->log.informed_mean += (float)env->informed_steps / fish_steps;
            env->log.beacon_visits += (float)visits / (float)env->num_agents;
        }
        env->log.n += 1.0f;
        puf_reset(env);
        for (int i = 0; i < env->num_agents; i++) {
            env->agents[i].terminals[0] = 1.0f;
        }
    }
}

Vector2 world_to_screen(const Wef* env, Vec2 p) {
    Client* client = env->client;
    // Uniform scale (same as the body/pellet radii below), arena centred: upstream stretched
    // x and y separately, which distorts every non-square arena.
    float usable_width = client->window_width - 2.0f * client->margin;
    float usable_height = client->window_height - 2.0f * client->margin;
    float scale = fminf(usable_width / env->arena_size_x, usable_height / env->arena_size_y);
    float x0 = client->margin + 0.5f * (usable_width - env->arena_size_x * scale);
    float y0 = client->window_height - client->margin -
        0.5f * (usable_height - env->arena_size_y * scale);
    return (Vector2){x0 + p.x * scale, y0 - p.y * scale};
}

static Color wef_lerp_color(Color a, Color b, float t) {
    t = clamp(t, 0.0f, 1.0f);
    return (Color){
        (unsigned char)((float)a.r + ((float)b.r - (float)a.r) * t),
        (unsigned char)((float)a.g + ((float)b.g - (float)a.g) * t),
        (unsigned char)((float)a.b + ((float)b.b - (float)a.b) * t),
        (unsigned char)((float)a.a + ((float)b.a - (float)a.a) * t),
    };
}

// Map |E| in V/m → color; colormap is defined in V/cm (yellow → red).
static Color wef_color_from_field(float strength_vm) {
    float strength_vcm = strength_vm * 0.01f;  // V/m → V/cm
    float log_s = log10f(fmaxf(strength_vcm, 1e-20f));
    float t = clamp(
        (log_s - WEF_FIELD_LOG_LO) / (WEF_FIELD_LOG_HI - WEF_FIELD_LOG_LO),
        0.0f,
        1.0f
    );
    return wef_lerp_color(WEF_COLOR_FIELD_WEAK, WEF_COLOR_FIELD_STRONG, t);
}

// Unit-direction arrow in screen space.
static void wef_draw_field_arrow(Vector2 base, float ux, float uy, float len, Color color) {
    Vector2 tip = {base.x + ux * len, base.y + uy * len};
    Vector2 wing = {base.x + ux * len * 0.75f, base.y + uy * len * 0.75f};
    float nx = -uy * len * 0.15f;
    float ny = ux * len * 0.15f;
    DrawLineV(base, tip, color);
    DrawLineV((Vector2){wing.x + nx, wing.y + ny}, tip, color);
    DrawLineV((Vector2){wing.x - nx, wing.y - ny}, tip, color);
}

// Compact V/cm color bar for the WEF log range.
static void wef_draw_field_colorbar(int win_w, int win_h) {
    const int bar_h = 120;
    const int bar_x0 = win_w - 18;
    const int bar_x1 = win_w - 10;
    const int bar_y1 = win_h - 14;
    const int bar_y0 = bar_y1 - bar_h;
    for (int i = 0; i < bar_h; i++) {
        float t = (float)i / (float)(bar_h - 1);
        float log_s = WEF_FIELD_LOG_LO + t * (WEF_FIELD_LOG_HI - WEF_FIELD_LOG_LO);
        // Color map expects V/m; convert V/cm → V/m (*100).
        Color c = wef_color_from_field(powf(10.0f, log_s) * 100.0f);
        DrawLine(bar_x0, bar_y1 - i, bar_x1, bar_y1 - i, c);
    }
    DrawRectangleLines(bar_x0 - 1, bar_y0, bar_x1 - bar_x0 + 2, bar_h, WEF_COLOR_MIDGRAY);
    DrawText("V/cm", win_w - 42, bar_y0 - 12, 10, WEF_COLOR_MIDGRAY);
    DrawText("1e-3", win_w - 48, bar_y0 + 2, 10, WEF_COLOR_MIDGRAY);
    DrawText("1e-7", win_w - 48, bar_y1 - 10, 10, WEF_COLOR_MIDGRAY);
}

void puf_render(Wef* env) {
    if (env->client == NULL) {
        Client* client = (Client*)calloc(1, sizeof(Client));
        client->window_width = 900;
        client->window_height = 900;
        client->margin = 55;
        client->show_field = env->render_field != 0;
        client->show_sensors = true;
        InitWindow(client->window_width, client->window_height, "Weakly Electric fish");
        SetTargetFPS(60);
        HideCursor();
        // Deterministic recording: WEF_VIDEO_OUT=clip.mp4 pipes one frame per env step to
        // ffmpeg (WEF_VIDEO_FPS frames/s, default 30) for the first episode, unthrottled;
        // WEF_VIDEO_EXIT=1 exits once the file is written. Screen grabbing instead samples
        // on its own clock and duplicates / drops steps (jerky motion).
        const char* video_out = getenv("WEF_VIDEO_OUT");
        if (video_out != NULL && video_out[0] != '\0') {
            const char* fps_env = getenv("WEF_VIDEO_FPS");
            int fps = fps_env != NULL && atoi(fps_env) > 0 ? atoi(fps_env) : 30;
            char cmd[4600];
            snprintf(cmd, sizeof(cmd),
                "ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgba -s %dx%d -r %d -i - "
                "-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p '%s'",
                GetRenderWidth(), GetRenderHeight(), fps, video_out);
            client->video = popen(cmd, "w");
            assert(client->video != NULL && "WEF_VIDEO_OUT: could not start ffmpeg");
            SetTargetFPS(0);
        }
        env->client = client;
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }
    // Hold Left Shift + WASD/arrows; Space bites. Skip F/S viz toggles while driving.
    if (IsWindowReady() && IsKeyDown(KEY_LEFT_SHIFT)) {
        float* a = env->agents[0].actions;
        a[0] = 0.0f;
        a[1] = 0.0f;
        a[2] = 1.0f;
        a[3] = -1.0f;
        if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
            a[0] = 1.0f;
        } else if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
            a[0] = -1.0f;
        }
        if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
            a[1] = -1.0f;
        } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
            a[1] = 1.0f;
        }
        if (IsKeyDown(KEY_SPACE)) {
            a[3] = 1.0f;
        }
    } else {
        if (IsKeyPressed(KEY_F)) {
            env->client->show_field = !env->client->show_field;
        }
        if (IsKeyPressed(KEY_S)) {
            env->client->show_sensors = !env->client->show_sensors;
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        Trace* trace = &env->client->traces[i];
        if (env->agents[i].terminals[0]) {
            trace->index = 0;
            trace->count = 0;
            if (i == 0 && env->client->video != NULL && env->client->video_frames > 0) {
                pclose(env->client->video);  // waits for ffmpeg to finish the file
                env->client->video = NULL;
                if (getenv("WEF_VIDEO_EXIT") != NULL) {
                    exit(0);
                }
            }
            if (i == 0) {
                for (int k = 0; k < MAX_AGENTS; k++) {
                    env->client->bites_given[k] = 0;
                    env->client->bites_taken[k] = 0;
                    env->client->bite_flash[k] = 0;
                }
                env->client->bites_total = 0;
            }
        }
        trace->pos[trace->index] = env->fish[i].pos;
        trace->index = (trace->index + 1) % TRACE_LENGTH;
        if (trace->count < TRACE_LENGTH) {
            trace->count++;
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        Client* c = env->client;
        c->bite_flash[i] -= c->bite_flash[i] > 0;
        int v = env->fish[i].bite_victim;
        if (v >= 0) {
            c->bite_flash[v] = WEF_BITE_FLASH_FRAMES;
            c->bite_from[v] = i;
            c->bites_given[i]++;
            c->bites_taken[v]++;
            c->bites_total++;
        }
    }

    BeginDrawing();
    ClearBackground(WEF_COLOR_BG);

    Vector2 arena_min = world_to_screen(env, (Vec2){0.0f, env->arena_size_y});
    Vector2 arena_max = world_to_screen(env, (Vec2){env->arena_size_x, 0.0f});
    DrawRectangleRec(
        (Rectangle){
            arena_min.x, arena_min.y,
            arena_max.x - arena_min.x, arena_max.y - arena_min.y
        },
        WEF_COLOR_BG
    );

    if (env->client->show_field) {
        Mono mono[WEF_MAX_MONO];
        Dipole induced[WEF_MAX_DIP];
        Dipole intrinsic[WEF_MAX_DIP];
        int n_mono = 0;
        int n_ind = 0;
        int n_intr = 0;
        for (int a = 0; a < env->num_agents; a++) {
            for (int p = 0; p < 2; p++) {
                mono[n_mono++] = (Mono){
                    to_m(env->fish[a].eod_pos[p]), env->fish[a].eod_charge[p]
                };
            }
            induced[n_ind++] = (Dipole){
                to_m(env->fish[a].pos), env->fish[a].induced_moment
            };
            intrinsic[n_intr++] = (Dipole){
                to_m(env->fish[a].pos), env->fish[a].intrinsic_moment
            };
        }
        for (int f = 0; f < env->num_food; f++) {
            if (!env->food[f].active) {
                continue;
            }
            induced[n_ind++] = (Dipole){
                to_m(env->food[f].pos), env->food[f].induced_moment
            };
            intrinsic[n_intr++] = (Dipole){
                to_m(env->food[f].pos), env->food[f].intrinsic_moment
            };
        }
        int n_food_viz = n_ind - env->num_agents;
        for (int w = 0; w < env->waste_max; w++) {
            if (!env->waste[w].active) {
                continue;
            }
            induced[n_ind++] = (Dipole){
                to_m(env->waste[w].pos), env->waste[w].induced_moment
            };
        }

        // Local vector field around each fish (fixed-length arrows, strength → color).
        const int columns = 36;
        const int rows = 36;
        float radius_squared =
            env->electric_field_radius_cm * env->electric_field_radius_cm;
        const float arrow_len = 10.0f;
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                Vec2 pos = {
                    env->arena_size_x * (column + 0.5f) / columns,
                    env->arena_size_y * (row + 0.5f) / rows,
                };
                bool near_fish = false;
                for (int i = 0; i < env->num_agents; i++) {
                    float dx = pos.x - env->fish[i].pos.x;
                    float dy = pos.y - env->fish[i].pos.y;
                    if (dx * dx + dy * dy <= radius_squared) {
                        near_fish = true;
                        break;
                    }
                }
                if (!near_fish) {
                    continue;
                }

                // Viz: knollen agent range for EODs; morm food range for food dips.
                Vec2 f1 = measure_field(
                    env, pos, mono, n_mono, induced, n_ind, env->num_agents,
                    n_food_viz,
                    KNOLLEN_AGENT_RANGE_CM, MORM_FOOD_RANGE_CM, env->waste_sense_range_cm,
                    env->reflection_wall_range_cm
                );
                Vec2 f2 = measure_field(
                    env, pos, NULL, 0, intrinsic, n_intr, env->num_agents,
                    n_intr - env->num_agents,
                    KNOLLEN_AGENT_RANGE_CM, AMP_FOOD_RANGE_CM, 0.0f,
                    env->reflection_wall_range_cm
                );
                float fx = f1.x + f2.x;
                float fy = f1.y + f2.y;
                float strength = sqrtf(fx * fx + fy * fy);
                if (strength <= 1e-20f) {
                    continue;
                }
                // Screen-space unit direction
                float ux = fx / strength;
                float uy = -(fy / strength);
                Vector2 base = world_to_screen(env, pos);
                // Center arrow on sample point (vector-field look).
                Vector2 mid = {
                    base.x - ux * arrow_len * 0.5f,
                    base.y - uy * arrow_len * 0.5f,
                };
                Color arrow = wef_color_from_field(strength);
                arrow.a = (unsigned char)((float)arrow.a * clamp(env->render_field_alpha, 0.0f, 1.0f));
                wef_draw_field_arrow(mid, ux, uy, arrow_len, arrow);
            }
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        Trace* trace = &env->client->traces[i];
        for (int j = 0; j < trace->count - 1; j++) {
            int current =
                (trace->index - j - 1 + TRACE_LENGTH) % TRACE_LENGTH;
            int previous =
                (trace->index - j - 2 + TRACE_LENGTH) % TRACE_LENGTH;
            float alpha =
                0.55f * (float)(trace->count - j) / (float)trace->count;
            DrawLineEx(
                world_to_screen(env, trace->pos[current]),
                world_to_screen(env, trace->pos[previous]),
                2.0f, ColorAlpha(WEF_COLOR_FISH, alpha)
            );
        }
    }

    Client* client = env->client;
    float scale = fminf(
        (client->window_width - 2.0f * client->margin) / env->arena_size_x,
        (client->window_height - 2.0f * client->margin) / env->arena_size_y
    );
    for (int i = 0; i < env->num_food; i++) {
        if (!env->food[i].active) {
            continue;
        }
        Vector2 position = world_to_screen(env, env->food[i].pos);
        float radius = fmaxf(2.5f, FOOD_RADIUS_CM * scale);
        if (env->allelo) {
            // AH: type by colour (A = today's pellet green, B magenta), ripeness by radius
            // (ripe = a full pellet, unripe = a small faint dot).
            Color c = env->food[i].type == 0 ? WEF_COLOR_FOOD : WEF_COLOR_FOOD_B;
            if (env->food[i].ripe) {
                DrawCircleV(position, fmaxf(radius, 1.6f * fmaxf(2.5f, env->food_radius_ripe_cm * scale)), c);
            } else {
                DrawCircleV(position, fmaxf(2.0f, 0.5f * radius), ColorAlpha(c, 0.55f));
            }
            continue;
        }
        DrawCircleV(position, radius, WEF_COLOR_FOOD);
    }
    if (env->cleanup) {
        // Zone outlines and waste items
        Vector2 s0 = world_to_screen(env, (Vec2){env->strip_cm, 0.0f});
        Vector2 s1 = world_to_screen(env, (Vec2){env->strip_cm, env->arena_size_y});
        DrawLineV(s0, s1, WEF_COLOR_MIDGRAY);
        Vector2 o0 = world_to_screen(env, (Vec2){env->arena_size_x - env->orchard_cm, 0.0f});
        Vector2 o1 = world_to_screen(env, (Vec2){env->arena_size_x - env->orchard_cm, env->arena_size_y});
        DrawLineV(o0, o1, WEF_COLOR_MIDGRAY);
        for (int i = 0; i < env->waste_max; i++) {
            if (!env->waste[i].active) {
                continue;
            }
            Vector2 position = world_to_screen(env, env->waste[i].pos);
            DrawCircleV(position, fmaxf(3.0f, env->waste_radius_cm * scale), WEF_COLOR_WASTE);
        }
    }
    if (env->inst_obs) {
        // Institution beacon: the read radius as a faint ring, the object as a dot in the rule's
        // colour (AH: the prescribed type; Commons: red while closed, green while open; grey in
        // the control arm, which has no rule).
        Vector2 bp = world_to_screen(env, env->inst_pos);
        Color rc = WEF_COLOR_MIDGRAY;
        const char* label = "BEACON: mute";
        if (env->inst_mode > 0) {
            rc = env->allelo ? (env->inst_type == 0 ? WEF_COLOR_FOOD : WEF_COLOR_FOOD_B)
                : (env->inst_type ? WEF_COLOR_BITE : WEF_COLOR_FOOD);
            label = env->allelo ? (env->inst_type == 0 ? "RULE: PLANT GREEN (A)" : "RULE: PLANT MAGENTA (B)")
                : (env->inst_type ? "SEASON: CLOSED" : "SEASON: OPEN");
        }
        float rr = env->inst_read_cm * scale;
        DrawRing(bp, rr - 1.5f, rr + 1.5f, 0.0f, 360.0f, 64, ColorAlpha(WEF_COLOR_INST, 0.6f));
        // the beacon: a large disc in the rule's colour with a white outline and a label
        float br = fmaxf(12.0f, env->inst_obj_radius_cm * scale);
        DrawCircleV(bp, br + 3.0f, WEF_COLOR_TEXT);
        DrawCircleV(bp, br, rc);
        DrawText(label, (int)(bp.x + br + 8.0f), (int)(bp.y - 9.0f), 18, WEF_COLOR_TEXT);
    }
    // Bites: red line biter -> victim and a fading red ring on the victim.
    for (int v = 0; v < env->num_agents; v++) {
        int f = env->client->bite_flash[v];
        if (f <= 0) {
            continue;
        }
        float a = (float)f / (float)WEF_BITE_FLASH_FRAMES;
        Vector2 vc = world_to_screen(env, env->fish[v].pos);
        Vector2 bc = world_to_screen(env, env->fish[env->client->bite_from[v]].pos);
        float r = BODY_RADIUS_CM * scale;
        DrawLineEx(bc, vc, 3.0f, ColorAlpha(WEF_COLOR_BITE, a));
        DrawRing(vc, r + 4.0f + 10.0f * (1.0f - a), r + 8.0f + 10.0f * (1.0f - a),
            0.0f, 360.0f, 32, ColorAlpha(WEF_COLOR_BITE, a));
    }
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->fish[i];
        Vector2 center = world_to_screen(env, agent->pos);
        float radius = BODY_RADIUS_CM * scale;
        if (env->allelo) {
            // taste: a faint body fill in the type colour (the body ring below is the same colour)
            Color tc = agent->taste == 0 ? WEF_COLOR_FOOD : WEF_COLOR_FOOD_B;
            DrawCircleV(center, radius - 1.0f, ColorAlpha(tc, 0.28f));
        }
        if (agent->freeze > 0) {
            DrawCircleV(center, radius, ColorAlpha(WEF_COLOR_MIDGRAY, 0.8f));  // frozen (sanction)
        } else if (env->allelo && agent->eat_cooldown > EAT_COOLDOWN_STEPS) {
            // planting hold: fill shrinks as the hold runs out
            float frac = (float)agent->eat_cooldown / (float)(env->plant_steps > 0 ? env->plant_steps : 1);
            DrawCircleV(center, radius * fmaxf(0.3f, fminf(1.0f, frac)), ColorAlpha(WEF_COLOR_HOLD, 0.8f));
        }
        if (agent->mark > 0) {
            // violation mark: a thick white ring with a dark outline outside the body (the taste ring
            // inside the body is green / magenta, so the mark is unmistakable), fading as it runs out
            float a = 0.45f + 0.55f * (float)agent->mark / (float)(env->inst_mark_steps > 0 ? env->inst_mark_steps : 1);
            DrawRing(center, radius + 3.0f, radius + 9.0f, 0.0f, 360.0f, 32, ColorAlpha((Color){0, 0, 0, 255}, a));
            DrawRing(center, radius + 4.5f, radius + 7.5f, 0.0f, 360.0f, 32, ColorAlpha(WEF_COLOR_MARK, a));
        }
        if (env->inst_mode > 0 && agent->informed) {
            // informed fish: a small dot in the rule colour it last read, at the tail
            Color ic = agent->inst_val > 0.0f ? (env->allelo ? WEF_COLOR_FOOD : WEF_COLOR_BITE)
                : agent->inst_val < 0.0f ? (env->allelo ? WEF_COLOR_FOOD_B : WEF_COLOR_FOOD) : WEF_COLOR_MIDGRAY;
            Vector2 tail = {center.x - cosf(agent->orientation) * radius * 1.4f,
                center.y + sinf(agent->orientation) * radius * 1.4f};
            DrawCircleV(tail, 3.0f, ic);
        }

        if (agent->emits_eod) {
            float pulse = radius + 5.0f +
                5.0f * sinf((float)env->tick * 0.18f);
            DrawCircleLines((int)center.x, (int)center.y, pulse,
                WEF_COLOR_FISH_PULSE);
        }

        float heading_x = cosf(agent->orientation);
        float heading_y = -sinf(agent->orientation);
        if (env->allelo) {
            // AH: the body ring IS the taste colour, thick, and the only ring on the body
            Color tc = agent->taste == 0 ? WEF_COLOR_FOOD : WEF_COLOR_FOOD_B;
            DrawRing(center, radius - 2.5f, radius + 2.0f, 0.0f, 360.0f, 32, ColorAlpha(tc, 1.0f));
        } else {
            DrawRing(center, radius - 1.5f, radius + 1.5f,
                0.0f, 360.0f, 32, WEF_COLOR_FISH);
        }
        Vector2 nose = {
            center.x + heading_x * radius * 1.35f,
            center.y + heading_y * radius * 1.35f,
        };
        DrawLineEx(center, nose, 3.0f, WEF_COLOR_FISH);

        Vector2 positive = world_to_screen(env, agent->eod_pos[0]);
        Vector2 negative = world_to_screen(env, agent->eod_pos[1]);
        DrawCircleV(positive, 3.0f, WEF_COLOR_EOD_POS);
        DrawCircleV(negative, 3.0f, WEF_COLOR_EOD_NEG);

        if (env->client->show_sensors) {
            for (int sensor_idx = 0; sensor_idx < NUM_KNOLLEN;
                    sensor_idx++) {
                Sensor w = sensor_world(&g_knollen[sensor_idx], agent);
                DrawCircleV(
                    world_to_screen(env, w.p),
                    1.5f, WEF_COLOR_SENSOR
                );
            }
        }
        DrawText(env->allelo ? TextFormat("%d%c", i + 1, agent->taste == 0 ? 'A' : 'B') : TextFormat("%d", i + 1),
            (int)(center.x + radius + 4), (int)(center.y - radius), 16,
            WEF_COLOR_TEXT);
        if (env->client->bites_given[i] > 0 || env->client->bites_taken[i] > 0) {
            const char* tally = TextFormat("bit %d / bitten %d",
                env->client->bites_given[i], env->client->bites_taken[i]);
            int w = MeasureText(tally, 12);
            int tx = (int)(center.x + radius + 4);
            if (tx + w > env->client->window_width - 4) {
                tx = (int)(center.x - radius - 4) - w;  // near the right edge: label on the left
            }
            DrawText(tally, tx, (int)(center.y - radius) + 16, 12, WEF_COLOR_BITE);
        }
    }

    DrawRectangleLinesEx(
        (Rectangle){
            arena_min.x, arena_min.y,
            arena_max.x - arena_min.x, arena_max.y - arena_min.y
        },
        2.0f, WEF_COLOR_MIDGRAY
    );
    DrawText("Weakly electric fish", 20, 14, 24, WEF_COLOR_TEXT);
    int active_eods = 0;
    for (int i = 0; i < env->num_agents; i++) {
        active_eods += env->fish[i].emits_eod ? 1 : 0;
    }
    {
        const char* top = TextFormat("step %d   active EODs %d/%d   bites %d",
            env->tick, active_eods, env->num_agents, env->client->bites_total);
        DrawText(top, env->client->window_width - 20 - MeasureText(top, 18), 18, 18, WEF_COLOR_MIDGRAY);
    }
    if (env->inst_mode > 0) {
        int marked = 0;
        int informed = 0;
        for (int i = 0; i < env->num_agents; i++) {
            marked += env->fish[i].mark > 0;
            informed += env->fish[i].informed;
        }
        const char* rule = env->allelo ? (env->inst_type == 0 ? "rule: plant A" : "rule: plant B")
            : (env->inst_type ? "season: CLOSED" : "season: open");
        const char* line = TextFormat("%s   informed %d/%d   violations %d   marked %d   zaps on marked %d/%d",
            rule, informed, env->num_agents, env->violations, marked, env->zaps_on_marked, env->bites);
        DrawText(line, 20, 44, 18, WEF_COLOR_INST);
        DrawText("beacon disc = rule colour | white ring = marked | tail dot = rule read | grey = frozen | yellow = planting hold",
            20, 66, 14, WEF_COLOR_MIDGRAY);
    }
    if (env->allelo) {
        const char* status = TextFormat("bushes A %d  B %d   ripe %d   eaten %d   plantings %d   zaps %d",
            env->n_type[0], env->n_type[1], env->n_ripe, env->food_eaten, env->plantings, env->bites);
        DrawText(status, env->client->window_width - 70 - MeasureText(status, 18),
            env->client->window_height - 32, 18, WEF_COLOR_MIDGRAY);
    } else if (env->cleanup) {
        // Longer status line: draw it right-aligned so it clears the field-radius text.
        const char* status = TextFormat("food %d active (%d eaten)   waste %d/%d   cleans %d",
            env->food_active, env->food_eaten, env->waste_active, env->waste_max, env->cleans);
        DrawText(status, env->client->window_width - 70 - MeasureText(status, 18),
            env->client->window_height - 32, 18, WEF_COLOR_MIDGRAY);
    } else if (env->regrows) {
        const char* status = TextFormat("food %d active (%d eaten, %d regrown)",
            env->food_active, env->food_eaten, env->regrown);
        DrawText(status, env->client->window_width - 70 - MeasureText(status, 18),
            env->client->window_height - 32, 18, WEF_COLOR_MIDGRAY);
    } else {
        DrawText(TextFormat("food %d/%d", env->food_eaten, env->num_food),
            20, env->client->window_height - 32, 18, WEF_COLOR_MIDGRAY);
    }
    DrawText(TextFormat("field radius %.0f cm", env->electric_field_radius_cm),
        env->cleanup || env->regrows || env->allelo ? 20 : 180,
        env->client->window_height - 32, 18, WEF_COLOR_MIDGRAY);

    if (env->client->show_field) {
        wef_draw_field_colorbar(
            env->client->window_width, env->client->window_height
        );
    }
    if (env->client->video != NULL) {
        rlDrawRenderBatchActive();
        int w = GetRenderWidth();
        int h = GetRenderHeight();
        unsigned char* pixels = rlReadScreenPixels(w, h);
        fwrite(pixels, 1, (size_t)w * (size_t)h * 4, env->client->video);
        free(pixels);
        env->client->video_frames++;
    }
    EndDrawing();
    puf_web_vsync();
}

void puf_close(Wef* env) {
    if (env->trace_file != NULL) {
        fclose(env->trace_file);
        env->trace_file = NULL;
    }
    if (env->obj_file != NULL) {
        fclose(env->obj_file);
        env->obj_file = NULL;
    }
    if (env->client) {
        if (IsWindowReady()) {
            CloseWindow();
        }
        free(env->client);
    }
}

// Optional [env] key: value if present in default.ini/wef.ini, else `fallback`.
// (dict_get exits on a missing key; the cleanup keys must not be required.)
double wef_cfg(Dict* kwargs, const char* key, double fallback) {
    DictItem* item = dict_find(kwargs, key);
    return item ? item->value : fallback;
}

// Every [env] key the env reads (plus default.ini's shared "dr"). The trainer and the CPU
// harness accept any --env.key, and wef_cfg falls back to a default for a missing key, so a
// misspelled key would silently run the default: reject unknown keys instead.
static const char* WEF_ENV_KEYS[] = {
    "dr", "num_agents", "num_bots", "min_arena_width", "max_arena_width", "min_arena_height",
    "max_arena_height", "food_distribution", "num_food", "patch_radius", "patch_radius_std",
    "patch_density", "electric_field_radius", "reflection_wall_range", "episode_length",
    "cleanup", "strip_cm", "orchard_cm", "spawn_band", "waste_max", "waste_start",
    "waste_spawn_p", "waste_spawn_delay", "waste_theta", "waste_radius_cm", "waste_contrast",
    "waste_sense_range_cm", "regrow_p_max", "regrow_mode", "regrow_allee", "season_steps",
    "season_growth", "regrow_radius_cm", "food_start", "clean_priority", "clean_radius_cm",
    "clean_max_items", "clean_cooldown_steps", "clean_reward", "clean_reward_anneal_steps",
    "bitten_freeze_steps", "bitten_reward", "bite_reward", "eod_cost", "reward_share",
    "proximity_shaping", "obs_extra", "size_min", "size_max", "desync_first_episode",
    "policy1_agents", "no_clean_agents", "roles", "bot_shift_steps", "bot_oracle", "bot_theta",
    "render_field", "render_field_alpha", "sustain_frac", "num_patches", "regrow_in_patches",
    "bot_camp",
    // Allelopathic Harvest (docs/wef-allelopathic-harvest-v0-design.md 4.1)
    "allelo", "ripen_lin", "ripen_cubic", "ripen_pow", "ripen_min_steps", "start_frac_a",
    "start_ripe_frac", "food_contrast_a", "food_contrast_b", "food_radius_unripe_cm",
    "food_radius_ripe_cm", "unripe_intrinsic", "ripe_intrinsic", "taste_match", "taste_other",
    "taste_other_a", "taste_other_b", "taste_split", "taste_n_a", "size_a_min", "size_a_max",
    "size_b_min", "size_b_max", "size_speed_exp", "plant_mode", "plant_split", "plant_bin_order",
    "plant_priority", "plant_radius_cm", "plant_steps", "plant_cooldown_steps", "plant_reward",
    "plant_reward_anneal_steps", "zap_cooldown_steps", "zap_steps", "conv_theta",
    "bot_plant_theta", "bot_plant_max", "bot_plant_frac",
    // Institutions (docs/institutions-plan.md section 4)
    "inst_mode", "inst_obs", "inst_fixed_type", "inst_theta", "inst_hyst", "inst_flip_steps",
    "inst_mark_steps", "mark_zap_reward", "mark_zap_cooldown", "inst_pos_random", "inst_read_cm",
    "inst_obj_radius_cm", "inst_contrast", "inst_latch", "mark_freeze_steps",
};

static void wef_check_keys(Dict* kwargs) {
    int n_known = (int)(sizeof(WEF_ENV_KEYS) / sizeof(WEF_ENV_KEYS[0]));
    for (int i = 0; i < kwargs->size; i++) {
        bool known = false;
        for (int k = 0; k < n_known && !known; k++) {
            known = strcmp(kwargs->items[i].key, WEF_ENV_KEYS[k]) == 0;
        }
        if (!known) {
            fprintf(stderr, "wef: unknown [env] key '%s' (misspelled? see config/wef.ini)\n",
                kwargs->items[i].key);
            exit(1);
        }
    }
}

void puf_init(Env* env, Dict* kwargs) {
    wef_check_keys(kwargs);
    env->num_agents = dict_get(kwargs, "num_agents");
    assert(env->num_agents > 0 && env->num_agents <= MAX_AGENTS);
    env->min_arena_size_x = dict_get(kwargs, "min_arena_width");
    env->min_arena_size_y = dict_get(kwargs, "min_arena_height");
    env->max_arena_size_x = dict_get(kwargs, "max_arena_width");
    env->max_arena_size_y = dict_get(kwargs, "max_arena_height");
    env->arena_size_x = env->min_arena_size_x;
    env->arena_size_y = env->min_arena_size_y;
    env->food_distribution = (FoodDistribution)dict_get(kwargs, "food_distribution");
    env->configured_num_food = dict_get(kwargs, "num_food");
    assert(env->configured_num_food > 0 && env->configured_num_food <= MAX_FOOD);
    env->patch_radius_cm = dict_get(kwargs, "patch_radius");
    env->patch_radius_std_cm = dict_get(kwargs, "patch_radius_std");
    env->patch_density = dict_get(kwargs, "patch_density");
    env->electric_field_radius_cm = dict_get(kwargs, "electric_field_radius");
    env->reflection_wall_range_cm = dict_get(kwargs, "reflection_wall_range");
    env->episode_length = dict_get(kwargs, "episode_length");

    // Cleanup extension (docs/wef-cleanup-v0-design.md section 8). Defaults = baseline.
    env->cleanup = wef_cfg(kwargs, "cleanup", 0);
    env->strip_cm = wef_cfg(kwargs, "strip_cm", 0);
    env->orchard_cm = wef_cfg(kwargs, "orchard_cm", 0);
    env->spawn_band = wef_cfg(kwargs, "spawn_band", 0);
    env->waste_max = wef_cfg(kwargs, "waste_max", 0);
    env->waste_start = wef_cfg(kwargs, "waste_start", 0);
    env->waste_spawn_p = wef_cfg(kwargs, "waste_spawn_p", 0);
    env->waste_spawn_delay = wef_cfg(kwargs, "waste_spawn_delay", 0);
    env->waste_theta = wef_cfg(kwargs, "waste_theta", 0.4);
    env->waste_radius_cm = wef_cfg(kwargs, "waste_radius_cm", 0.5);
    env->waste_contrast = wef_cfg(kwargs, "waste_contrast", 1.0);
    env->waste_sense_range_cm = wef_cfg(kwargs, "waste_sense_range_cm", 10.0);
    env->regrow_p_max = wef_cfg(kwargs, "regrow_p_max", 0);
    env->regrow_mode = wef_cfg(kwargs, "regrow_mode", 0);
    env->regrow_radius_cm = wef_cfg(kwargs, "regrow_radius_cm", 4.0);
    env->food_start = wef_cfg(kwargs, "food_start", -1);
    env->clean_priority = wef_cfg(kwargs, "clean_priority", 0);
    env->clean_radius_cm = wef_cfg(kwargs, "clean_radius_cm", BITING_RADIUS_CM);
    env->clean_max_items = wef_cfg(kwargs, "clean_max_items", 1);
    env->clean_cooldown_steps = wef_cfg(kwargs, "clean_cooldown_steps", BITE_COOLDOWN_STEPS);
    env->clean_reward = wef_cfg(kwargs, "clean_reward", 0);
    env->clean_reward_anneal_steps = wef_cfg(kwargs, "clean_reward_anneal_steps", 0);
    env->bitten_freeze_steps = wef_cfg(kwargs, "bitten_freeze_steps", 0);
    env->bitten_reward = wef_cfg(kwargs, "bitten_reward", BITTEN_REWARD);
    env->bite_reward = wef_cfg(kwargs, "bite_reward", BITE_REWARD);
    env->eod_cost = wef_cfg(kwargs, "eod_cost", 0);
    env->reward_share = wef_cfg(kwargs, "reward_share", 0);
    env->proximity_shaping = wef_cfg(kwargs, "proximity_shaping", PROXIMITY_SHAPING_REWARD);
    env->obs_extra = wef_cfg(kwargs, "obs_extra", 0);
    env->size_min = wef_cfg(kwargs, "size_min", 0.0);
    env->size_max = wef_cfg(kwargs, "size_max", 1.0);
    env->desync_first_episode = wef_cfg(kwargs, "desync_first_episode", 0);
    env->policy1_agents = wef_cfg(kwargs, "policy1_agents", 0);
    env->no_clean_agents = wef_cfg(kwargs, "no_clean_agents", 0);
    env->bot_shift_steps = wef_cfg(kwargs, "bot_shift_steps", 256);
    env->bot_oracle = wef_cfg(kwargs, "bot_oracle", 0);
    env->bot_theta = wef_cfg(kwargs, "bot_theta", 0.5);
    env->regrow_allee = wef_cfg(kwargs, "regrow_allee", 0.0);
    env->render_field = wef_cfg(kwargs, "render_field", 1);
    env->render_field_alpha = wef_cfg(kwargs, "render_field_alpha", 1.0);
    env->season_steps = wef_cfg(kwargs, "season_steps", 0);
    env->season_growth = wef_cfg(kwargs, "season_growth", 2.0);
    env->sustain_frac = wef_cfg(kwargs, "sustain_frac", 0.5);
    env->num_patches = wef_cfg(kwargs, "num_patches", 0);
    env->regrow_in_patches = wef_cfg(kwargs, "regrow_in_patches", 0);
    env->bot_camp = wef_cfg(kwargs, "bot_camp", 0);
    // Allelopathic Harvest (section 4.1). Defaults reproduce upstream.
    env->allelo = wef_cfg(kwargs, "allelo", 0);
    env->ripen_lin = wef_cfg(kwargs, "ripen_lin", 0);
    env->ripen_cubic = wef_cfg(kwargs, "ripen_cubic", 0);
    env->ripen_pow = wef_cfg(kwargs, "ripen_pow", 3.0);
    env->ripen_min_steps = wef_cfg(kwargs, "ripen_min_steps", 0);
    env->start_frac_a = wef_cfg(kwargs, "start_frac_a", 0.5);
    env->start_ripe_frac = wef_cfg(kwargs, "start_ripe_frac", 0);
    env->food_contrast_a = wef_cfg(kwargs, "food_contrast_a", CONDUCTOR_CONTRAST);
    env->food_contrast_b = wef_cfg(kwargs, "food_contrast_b", 0.5);
    env->food_radius_unripe_cm = wef_cfg(kwargs, "food_radius_unripe_cm", FOOD_RADIUS_CM);
    env->food_radius_ripe_cm = wef_cfg(kwargs, "food_radius_ripe_cm", FOOD_RADIUS_CM);
    env->unripe_intrinsic = wef_cfg(kwargs, "unripe_intrinsic", 1.0);
    env->ripe_intrinsic = wef_cfg(kwargs, "ripe_intrinsic", 1.0);
    env->taste_match = wef_cfg(kwargs, "taste_match", EAT_REWARD);
    env->taste_other = wef_cfg(kwargs, "taste_other", EAT_REWARD);
    env->taste_other_a = wef_cfg(kwargs, "taste_other_a", -1);
    env->taste_other_b = wef_cfg(kwargs, "taste_other_b", -1);
    env->taste_split = wef_cfg(kwargs, "taste_split", 0.5);
    env->taste_n_a = wef_cfg(kwargs, "taste_n_a", -1);
    env->size_a_min = wef_cfg(kwargs, "size_a_min", 0.5);
    env->size_a_max = wef_cfg(kwargs, "size_a_max", 0.5);
    env->size_b_min = wef_cfg(kwargs, "size_b_min", 0.5);
    env->size_b_max = wef_cfg(kwargs, "size_b_max", 0.5);
    env->size_speed_exp = wef_cfg(kwargs, "size_speed_exp", SIZE_SPEED_EXPONENT);
    env->plant_mode = wef_cfg(kwargs, "plant_mode", 1);
    env->plant_split = wef_cfg(kwargs, "plant_split", 0.6745);
    env->plant_bin_order = wef_cfg(kwargs, "plant_bin_order", 0);
    env->plant_priority = wef_cfg(kwargs, "plant_priority", 0);
    env->plant_radius_cm = wef_cfg(kwargs, "plant_radius_cm", BITING_RADIUS_CM);
    env->plant_steps = wef_cfg(kwargs, "plant_steps", 0);
    env->plant_cooldown_steps = wef_cfg(kwargs, "plant_cooldown_steps", BITE_COOLDOWN_STEPS);
    env->plant_reward = wef_cfg(kwargs, "plant_reward", 0);
    env->plant_reward_anneal_steps = wef_cfg(kwargs, "plant_reward_anneal_steps", 0);
    env->zap_cooldown_steps = wef_cfg(kwargs, "zap_cooldown_steps", BITE_COOLDOWN_STEPS);
    env->zap_steps = wef_cfg(kwargs, "zap_steps", 0);
    env->conv_theta = wef_cfg(kwargs, "conv_theta", 0.9);
    env->bot_plant_theta = wef_cfg(kwargs, "bot_plant_theta", 1.0);
    env->bot_plant_max = wef_cfg(kwargs, "bot_plant_max", 0);
    env->bot_plant_frac = wef_cfg(kwargs, "bot_plant_frac", 1.0);
    // Institutions (section 4). Defaults = no institution, no beacon, no extra slots filled.
    env->inst_mode = wef_cfg(kwargs, "inst_mode", 0);
    env->inst_obs = wef_cfg(kwargs, "inst_obs", 0);
    env->inst_fixed_type = wef_cfg(kwargs, "inst_fixed_type", -1);
    env->inst_theta = wef_cfg(kwargs, "inst_theta", 0.5);
    env->inst_hyst = wef_cfg(kwargs, "inst_hyst", 0.1);
    env->inst_flip_steps = wef_cfg(kwargs, "inst_flip_steps", 0);
    env->inst_mark_steps = wef_cfg(kwargs, "inst_mark_steps", 0);
    env->mark_zap_reward = wef_cfg(kwargs, "mark_zap_reward", 0);
    env->mark_zap_cooldown = wef_cfg(kwargs, "mark_zap_cooldown", -1);
    env->mark_freeze_steps = wef_cfg(kwargs, "mark_freeze_steps", -1);
    env->inst_pos_random = wef_cfg(kwargs, "inst_pos_random", 0);
    env->inst_read_cm = wef_cfg(kwargs, "inst_read_cm", 5.0);
    env->inst_obj_radius_cm = wef_cfg(kwargs, "inst_obj_radius_cm", 0.5);
    env->inst_contrast = wef_cfg(kwargs, "inst_contrast", 1.0);
    env->inst_latch = wef_cfg(kwargs, "inst_latch", 1);
    {
        // roles = r0,r1,r2,r3 (comma list; a scalar applies to slot 0 only)
        DictItem* item = dict_find(kwargs, "roles");
        for (int i = 0; i < MAX_AGENTS; i++) {
            env->roles[i] = 0;
            if (item != NULL) {
                if (item->len > 0 && i < item->len) {
                    env->roles[i] = (int)item->values[i];
                } else if (item->len == 0 && i == 0) {
                    env->roles[i] = (int)item->value;
                }
            }
        }
    }
    assert(env->bot_shift_steps >= MAX_AGENTS);
    for (int i = 0; i < MAX_AGENTS; i++) {
        assert(env->roles[i] >= 0 && env->roles[i] <= 14
            && "roles: 0 policy, 1 cleaner, 2 eater, 3 shift, 4 random, 5 sustainable eater, 6 dense-first eater, 7 stock-threshold eater, 8 planter-own, 9 planter-majority, 10 free-rider, 11 zapper-planter, 12 opportunistic planter, 13 complier, 14 enforcer");
        assert((env->allelo || env->roles[i] <= 7 || env->roles[i] >= 13) && "roles 8-12 need allelo = 1");
        assert((env->inst_mode > 0 || env->roles[i] < 13) && "roles 13-14 need inst_mode > 0");
    }
    assert(env->inst_mode >= 0 && env->inst_mode <= 3 && "inst_mode: 0 none, 1 public beacon, 2 private signals, 3 spurious rule");
    assert((env->inst_mode == 0 || env->inst_obs) && "inst_mode > 0 needs inst_obs = 1 (the beacon and the slots)");
#ifndef WEF_INST
    assert(env->inst_obs == 0 && "inst_obs needs the institution build (NVCC_EXTRA / CFLAGS -DWEF_INST)");
#endif
    assert((env->inst_mode == 0 || env->allelo || env->regrow_p_max > 0.0f || env->regrow_mode == 3)
        && "an institution needs Allelopathic Harvest (rule = the prescribed type) or a regrowth mode (rule = the closed season)");
    assert(env->inst_fixed_type >= -1 && env->inst_fixed_type <= 1);
    assert(env->inst_theta > 0.0f && env->inst_theta < 1.0f && env->inst_hyst >= 0.0f && env->inst_theta + env->inst_hyst <= 1.0f);
    assert((env->inst_mode != 3 || env->inst_flip_steps > 0) && "inst_mode 3 needs inst_flip_steps > 0");
    assert(env->inst_mark_steps >= 0 && env->mark_zap_cooldown >= -1 && env->mark_freeze_steps >= -1
        && env->inst_read_cm > 0.0f && env->inst_obj_radius_cm > 0.0f);
    assert((env->inst_pos_random == 0 || env->inst_pos_random == 1) && (env->inst_latch == 0 || env->inst_latch == 1));
    if (env->allelo) {
        assert(env->cleanup == 0 && env->regrow_mode == 0 && env->regrow_p_max == 0.0f
            && env->waste_max == 0 && env->food_start == -1
            && "allelo: no cleanup / regrowth / waste / food_start (every slot is a permanent bush)");
        assert(env->obs_extra != 1 && env->obs_extra != 2 && "allelo: obs_extra 1/2 are Cleanup cues");
        assert(env->ripen_lin >= 0.0f && env->ripen_cubic >= 0.0f && env->ripen_lin + env->ripen_cubic <= 1.0f);
        assert(env->ripen_min_steps >= 0);
        assert(env->start_frac_a >= 0.0f && env->start_frac_a <= 1.0f);
        assert(env->start_ripe_frac >= 0.0f && env->start_ripe_frac <= 1.0f);
        assert(env->taste_n_a >= -2 && env->taste_n_a <= env->num_agents
            && "taste_n_a: -2 per-episode split, -1 free draw, 0..num_agents A-tasting fish");
        if (env->taste_n_a != -1) {
            assert(env->size_a_min >= 0.0f && env->size_a_min <= env->size_a_max && env->size_a_max <= 1.0f);
            assert(env->size_b_min >= 0.0f && env->size_b_min <= env->size_b_max && env->size_b_max <= 1.0f);
            bool both_used = env->taste_n_a == -2
                || (env->taste_n_a > 0 && env->taste_n_a < env->num_agents);
            if (both_used) {
                float width_a = env->size_a_max - env->size_a_min;
                float width_b = env->size_b_max - env->size_b_min;
                float width = width_a > width_b ? width_a : width_b;
                assert(env->size_a_min - env->size_b_max >= width + 0.10f - 1e-6f
                    && "allelo: band A must sit above band B by >= max(width) + 0.10 (group label, section 2.4)");
            }
        }
        assert(env->plant_steps >= 0 && env->plant_cooldown_steps >= 0 && env->zap_cooldown_steps >= 0
            && env->zap_steps >= 0);
        assert(env->plant_split > 0.0f);
        assert(env->plant_mode >= 0 && env->plant_mode <= 2);
        assert(env->plant_priority >= 0 && env->plant_priority <= 2);
        assert(env->plant_bin_order == 0 || env->plant_bin_order == 1);
        assert(env->plant_radius_cm > 0.0f);
        assert(env->taste_other <= env->taste_match && env->taste_other >= 0.0f);
        assert((env->taste_other_a == -1.0f || (env->taste_other_a >= 0.0f && env->taste_other_a <= env->taste_match))
            && (env->taste_other_b == -1.0f || (env->taste_other_b >= 0.0f && env->taste_other_b <= env->taste_match))
            && "allelo: taste_other_a / taste_other_b are -1 (inherit taste_other) or in [0, taste_match]");
        assert(env->food_radius_unripe_cm > 0.0f && env->food_radius_ripe_cm > 0.0f);
        assert(env->ripe_intrinsic >= 0.0f && env->unripe_intrinsic >= 0.0f);
        assert(env->conv_theta > 0.0f && env->conv_theta <= 1.0f);
        assert(env->bot_plant_theta >= 0.0f && env->bot_plant_max >= 0
            && env->bot_plant_frac >= 0.0f && env->bot_plant_frac <= 1.0f);
    }
    assert((int)wef_cfg(kwargs, "num_bots", 0) == 0 && "num_bots is not supported by wef");
    assert(env->waste_max >= 0 && env->waste_max <= MAX_WASTE);
    if (env->waste_max == 0) {
        env->waste_start = 0;  // Harvest stage: no waste at all, whatever the preset says
    }
    assert(env->waste_start >= 0 && env->waste_start <= env->waste_max);
    assert(env->reward_share >= 0.0f && env->reward_share <= 1.0f);
    assert(env->waste_theta > 0.0f);
    assert(env->clean_max_items >= 1);
    assert(env->policy1_agents >= 0 && env->no_clean_agents >= 0
        && env->policy1_agents + env->no_clean_agents < env->num_agents);
    assert(env->food_start <= env->configured_num_food);
    assert(env->size_min <= env->size_max);
    if (env->cleanup) {
        assert(env->strip_cm + env->orchard_cm + 6.0f <= env->min_arena_size_x
            && "cleanup: strip + orchard must leave a >= 6 cm spawn band");
        assert(env->orchard_cm > 0.0f && "cleanup: orchard_cm must be > 0");
        assert((env->waste_max == 0 || env->strip_cm > 0.0f) && "cleanup: waste needs strip_cm > 0");
    } else {
        assert(env->waste_max == 0 && "waste needs cleanup = 1");
        assert((env->regrow_p_max == 0.0f || env->regrow_mode >= 1)
            && "regrowth without cleanup needs regrow_mode = 1 (Harvest), 2 (Commons) or 3 (seasonal)");
    }
    assert(env->regrow_mode >= 0 && env->regrow_mode <= 3);
    assert((env->regrow_mode != 3 || (env->season_steps > 0 && !env->cleanup))
        && "regrow_mode 3 (seasonal) needs season_steps > 0 and cleanup = 0");
    env->regrows = env->regrow_p_max > 0.0f || env->regrow_mode == 3;
    assert(env->num_patches >= 0 && env->num_patches <= MAX_PATCHES);
    assert((!env->regrow_in_patches || (env->regrow_mode >= 2 && env->food_distribution != FOOD_UNIFORM))
        && "regrow_in_patches needs a commons mode (2/3) and a patchy food layout");
    assert(env->regrow_radius_cm > 0.0f);

    // The trainer seeds env->rng with the env index before puf_init.
    env->env_id = (int)env->rng;
    // desync_first_episode: stagger the first episode end so fixed-length episodes
    // do not reset in lockstep across envs (rollouts would see one phase only).
    env->first_episode_len = env->episode_length;
    if (env->desync_first_episode) {
        int offset = (env->env_id * 397) % env->episode_length;
        env->first_episode_len = env->episode_length - offset;
    }
    const char* trace_dir = getenv("WEF_TRACE_DIR");
    if (trace_dir != NULL && trace_dir[0] != '\0') {
        char path[4096];
        bool v2 = env->cleanup || env->regrows;
        bool v4 = env->inst_mode > 0 || env->inst_obs;
        snprintf(path, sizeof(path),
            v4 ? "%s/env_%05d.v4.bin" : env->allelo ? "%s/env_%05d.v3.bin" : v2 ? "%s/env_%05d.v2.bin" : "%s/env_%05d.bin",
            trace_dir, env->env_id);
        env->trace_file = fopen(path, "wb");
        assert(env->trace_file != NULL && "WEF_TRACE_DIR must be an existing, writable dir");
        snprintf(path, sizeof(path), "%s/env_%05d.obj.bin", trace_dir, env->env_id);
        env->obj_file = fopen(path, "wb");
        assert(env->obj_file != NULL);
    }
    for (int i = 0; i < env->num_agents; i++) {
        // policy1_agents: last k fish on policy 1 (used by `match` eval only; training
        // with one policy forces every agent to policy 0 in env_setup).
        env->agents[i].policy = i >= env->num_agents - env->policy1_agents ? 1 : 0;
        env->agents[i].action_mask = NULL;
    }

    // Shared body-frame sensor rings (fixed body radius).
    float r = BODY_RADIUS_CM;
    float chin = PI_F / 3.0f;
    int num_chin = 10;
    int num_rest = NUM_MORMYROMASTS - num_chin;
    for (int s = 0; s < num_chin; s++) {
        float a = -0.5f * chin + chin * (float)s / (float)(num_chin - 1);
        float c = cosf(a);
        float sn = sinf(a);
        g_morm[s] = (Sensor){{c * r, sn * r}, {c, sn}};
    }
    for (int s = 0; s < num_rest; s++) {
        float a = 0.5f * chin
            + (2.0f * PI_F - chin) * (float)s / (float)num_rest;
        float c = cosf(a);
        float sn = sinf(a);
        g_morm[num_chin + s] = (Sensor){{c * r, sn * r}, {c, sn}};
    }
    for (int s = 0; s < NUM_AMPULLARY; s++) {
        float a = 2.0f * PI_F * (float)s / (float)NUM_AMPULLARY;
        float c = cosf(a);
        float sn = sinf(a);
        g_amp[s] = (Sensor){{c * r, sn * r}, {c, sn}};
    }
    for (int s = 0; s < NUM_KNOLLEN; s++) {
        float a = 2.0f * PI_F * (float)s / (float)NUM_KNOLLEN;
        float c = cosf(a);
        float sn = sinf(a);
        g_knollen[s] = (Sensor){{c * r, sn * r}, {c, sn}};
    }
}

// Allelopathic Harvest keys (section 5.1). Emitted in both builds; only their position differs.
static void wef_log_allelo(Log* log, Dict* out) {
    dict_set(out, "mono_frac", log->mono_frac);
    dict_set(out, "frac_a_final", log->frac_a_final);
    dict_set(out, "frac_a_mean", log->frac_a_mean);
    dict_set(out, "majority_frac_final", log->majority_frac_final);
    dict_set(out, "conv_c", log->conv_c);
    dict_set(out, "time_to_convention", log->time_to_convention);
    dict_set(out, "ripe_frac", log->ripe_frac);
    dict_set(out, "plantings", log->plantings);
    dict_set(out, "plantings_a", log->plantings_a);
    dict_set(out, "plantings_b", log->plantings_b);
    dict_set(out, "plant_own_frac", log->plant_own_frac);
    dict_set(out, "plant_own_frac_a", log->plant_own_frac_a);
    dict_set(out, "plant_own_frac_b", log->plant_own_frac_b);
    dict_set(out, "plant_major_frac", log->plant_major_frac);
    dict_set(out, "plant_proactive", log->plant_proactive);
    dict_set(out, "plant_gini", log->plant_gini);
    dict_set(out, "taste_a_n", log->taste_a_n);
    dict_set(out, "zaps_cross", log->zaps_cross);
    dict_set(out, "zaps_same", log->zaps_same);
}

// Institution keys (docs/institutions-plan.md 4-5). The first group is what the dashboard shows
// in the -DWEF_INST AH build; the tail lands in the [metrics] series only.
static void wef_log_inst(Log* log, Dict* out) {
    dict_set(out, "comply_frac", log->comply_frac);
    dict_set(out, "inst_agree_final", log->inst_agree_final);
    dict_set(out, "informed_frac", log->informed_frac);
    dict_set(out, "zaps_on_marked_share", log->zaps_on_marked_share);
    dict_set(out, "violations", log->violations);
}

static void wef_log_inst_tail(Log* log, Dict* out) {
    dict_set(out, "inst_type_a", log->inst_type_a);
    dict_set(out, "comply_informed", log->comply_informed);
    dict_set(out, "marked_frac", log->marked_frac);
    dict_set(out, "marked_exposure", log->marked_exposure);
    dict_set(out, "zaps_on_marked", log->zaps_on_marked);
    dict_set(out, "inst_agree_mean", log->inst_agree_mean);
    dict_set(out, "plantings_p", log->plantings_p);
    dict_set(out, "plantings_np", log->plantings_np);
    dict_set(out, "closed_frac", log->closed_frac);
    dict_set(out, "eats_closed", log->eats_closed);
    dict_set(out, "mark_zap_bounty", log->mark_zap_bounty);
    dict_set(out, "beacon_first_visit", log->beacon_first_visit);
    dict_set(out, "informed_mean", log->informed_mean);
    dict_set(out, "beacon_visits", log->beacon_visits);
}

// The rest of the AH keys (off the 30-key live dashboard in the AH build, in the [metrics] series).
static void wef_log_allelo_tail(Log* log, Dict* out) {
    dict_set(out, "mono_final", log->mono_final);
    dict_set(out, "ripened", log->ripened);
    dict_set(out, "plant_noop", log->plant_noop);
    dict_set(out, "plant_attempts", log->plant_attempts);
    dict_set(out, "eaten_match_frac", log->eaten_match_frac);
    dict_set(out, "taste_a_return", log->taste_a_return);
    dict_set(out, "taste_b_return", log->taste_b_return);
}

#ifdef WEF_AH_DASH
// Allelopathic Harvest dashboard order (section 5.1): the trainer shows the first 30 env/*
// keys. Set by the AH build line (NVCC_EXTRA="-DMAX_AGENTS=8 -DWEF_AH_DASH"); the default
// build keeps today's order below so the Cleanup / Commons keys stay on their dashboards.
void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "food_eaten_mean", log->food_eaten_mean);
#ifdef WEF_INST
    // institution build: the 5 institution keys take the dashboard places of plant_own_frac_a / _b,
    // plant_gini, taste_a_n and eod_rate (all still in the [metrics] series below)
    dict_set(out, "mono_frac", log->mono_frac);
    dict_set(out, "frac_a_final", log->frac_a_final);
    dict_set(out, "frac_a_mean", log->frac_a_mean);
    dict_set(out, "majority_frac_final", log->majority_frac_final);
    dict_set(out, "conv_c", log->conv_c);
    dict_set(out, "time_to_convention", log->time_to_convention);
    dict_set(out, "ripe_frac", log->ripe_frac);
    dict_set(out, "plantings", log->plantings);
    dict_set(out, "plantings_a", log->plantings_a);
    dict_set(out, "plantings_b", log->plantings_b);
    dict_set(out, "plant_own_frac", log->plant_own_frac);
    dict_set(out, "plant_major_frac", log->plant_major_frac);
    dict_set(out, "plant_proactive", log->plant_proactive);
    dict_set(out, "zaps_cross", log->zaps_cross);
    dict_set(out, "zaps_same", log->zaps_same);
    wef_log_inst(log, out);
    dict_set(out, "bites", log->bites);
    dict_set(out, "freezes", log->freezes);
    dict_set(out, "frozen_frac", log->frozen_frac);
    dict_set(out, "equality", log->equality);
    dict_set(out, "collective_food", log->collective_food);
    dict_set(out, "eod_rate", log->eod_rate);
    dict_set(out, "plant_own_frac_a", log->plant_own_frac_a);
    dict_set(out, "plant_own_frac_b", log->plant_own_frac_b);
    dict_set(out, "plant_gini", log->plant_gini);
    dict_set(out, "taste_a_n", log->taste_a_n);
    wef_log_inst_tail(log, out);
#else
    wef_log_allelo(log, out);
    dict_set(out, "bites", log->bites);
    dict_set(out, "freezes", log->freezes);
    dict_set(out, "frozen_frac", log->frozen_frac);
    dict_set(out, "equality", log->equality);
    dict_set(out, "collective_food", log->collective_food);
    dict_set(out, "eod_rate", log->eod_rate);
    wef_log_inst(log, out);
    wef_log_inst_tail(log, out);
#endif
    wef_log_allelo_tail(log, out);
    dict_set(out, "collisions_fish", log->collisions_fish);
    dict_set(out, "food_per_fish_area", log->food_per_fish_area);
    dict_set(out, "waste_frac", log->waste_frac);
    dict_set(out, "frac_open", log->frac_open);
    dict_set(out, "cleans", log->cleans);
    dict_set(out, "regrown", log->regrown);
    dict_set(out, "clean_gini", log->clean_gini);
    dict_set(out, "clean_max_share", log->clean_max_share);
    dict_set(out, "strip_frac", log->strip_frac);
    dict_set(out, "bites_in_strip", log->bites_in_strip);
    dict_set(out, "bites_on_eaters", log->bites_on_eaters);
    dict_set(out, "bites_on_defectors", log->bites_on_defectors);
    dict_set(out, "bites_at_risk", log->bites_at_risk);
    dict_set(out, "bites_top", log->bites_top);
    dict_set(out, "eater_frac", log->eater_frac);
    dict_set(out, "defector_frac", log->defector_frac);
    dict_set(out, "collapsed", log->collapsed);
    dict_set(out, "survival_frac", log->survival_frac);
    dict_set(out, "policy_0_score", log->policy_0_score);
    dict_set(out, "policy_1_score", log->policy_1_score);
    dict_set(out, "draw_rate", log->draw_rate);
    dict_set(out, "n", log->n);
}
#else
void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "food_eaten_mean", log->food_eaten_mean);
    dict_set(out, "eod_rate", log->eod_rate);
    dict_set(out, "collisions_fish", log->collisions_fish);
    dict_set(out, "bites", log->bites);
    dict_set(out, "food_per_fish_area", log->food_per_fish_area);
    dict_set(out, "collective_food", log->collective_food);
#ifdef WEF_INST
    wef_log_inst(log, out);  // 4-fish institution build (Commons arms): on the dashboard
#endif
    dict_set(out, "waste_frac", log->waste_frac);
    dict_set(out, "frac_open", log->frac_open);
    dict_set(out, "cleans", log->cleans);
    dict_set(out, "regrown", log->regrown);
    dict_set(out, "equality", log->equality);
    dict_set(out, "clean_gini", log->clean_gini);
    dict_set(out, "clean_max_share", log->clean_max_share);
    dict_set(out, "strip_frac", log->strip_frac);
    dict_set(out, "bites_in_strip", log->bites_in_strip);
    dict_set(out, "freezes", log->freezes);
    dict_set(out, "bites_on_eaters", log->bites_on_eaters);
    dict_set(out, "bites_on_defectors", log->bites_on_defectors);
    dict_set(out, "bites_at_risk", log->bites_at_risk);
    dict_set(out, "bites_top", log->bites_top);
    dict_set(out, "eater_frac", log->eater_frac);
    dict_set(out, "defector_frac", log->defector_frac);
    dict_set(out, "frozen_frac", log->frozen_frac);
    dict_set(out, "collapsed", log->collapsed);
    dict_set(out, "survival_frac", log->survival_frac);
    dict_set(out, "policy_0_score", log->policy_0_score);
    dict_set(out, "policy_1_score", log->policy_1_score);
    dict_set(out, "draw_rate", log->draw_rate);
    dict_set(out, "n", log->n);
    // AH keys appended after today's order (default build: dashboard order unchanged).
    wef_log_allelo(log, out);
    wef_log_allelo_tail(log, out);
#ifndef WEF_INST
    wef_log_inst(log, out);
#endif
    wef_log_inst_tail(log, out);
}
#endif
