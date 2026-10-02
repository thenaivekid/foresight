"""
prompts.py — THE single home for every prompt string in this project.

Why one file: `config.py` is settings (model id, dtype, thresholds, budgets) and
this is content (what we say to the model). Keeping prompt text here means one
place to edit, one place to diff, and no import cycle — the previous split had
config.py and task_prompts_all9.py importing each other, which forced a lazy
loader to break the cycle at runtime.

What lives here:
  * per-task controller ICL for ALL 9 OmniPro tasks -> TASK_CONTROLLER_PROMPTS
  * per-task writer prompts (probe-gate arm)        -> TASK_WRITER_PROMPTS
  * the generic fallback controller DSL             -> CONTROLLER_PROMPT_GENERIC
  * probe-gate + system templates                   -> SYSTEM_PROMPT, GOAL_QUESTION,
                                                       WRITER_PROMPT, WRITER_CUE
  * the (role, format, semantics) split per task    -> TASK_PROMPT_PARTS
    so the FORMAT half can be stripped mechanically once the schema-walk decoder
  owns structure).

NOTE ON AUDIO: this project ingests NO audio and evaluates the
`audio_dependency=none` subset only. Every block below is deliberately
VISUAL-ONLY — no audio, speech or sound references anywhere.
"""
from __future__ import annotations



# ---------------------------------------------------------------------------
# Task-specific controller ICL prompts.
# Each replaces the generic `controller_prompt` for one OmniPro task category
# (the adapter selects by sample.task). They teach the SAME control-JSON DSL but
# with a worked, task-concrete (Situation -> Control) example. Schema per tick:
#   seen, have_enough_info, event_time_s, answer, fps, next_check_s, question_for_next
# (There is no `new_event` field: dedup is code-owned via the ev0 event_time_s
#  identity rule.)
# The controller appends an "Already reported" conversation history + the final
# emit cue at runtime, so these strings END with the worked example.
# ---------------------------------------------------------------------------
# NOTE ON AUDIO: the OmniPro paper defines semantic_condition_alert as "audio-first",
# but our pipeline does NOT use audio at all right now — we monitor VISUAL + semantic
# cues only. This ICL therefore deliberately avoids any audio/speech references.
#
# Paper's task intent (baked into this prompt): SCA tests COMPREHENSION/JUDGMENT, not
# perception. The model must (1) understand the user's abstract intent, (2) reason about
# whether what's on screen actually MEETS it, (3) alert at each occurrence. Each response
# should state WHAT happened AND WHY it satisfies the condition, in under 25 words. In the
# eval, a temporal match only counts if the response is also content-correct (LLM-judged).
_SEMANTIC_CONDITION_ALERT = (
    "\nYou are the CONTROLLER of a live video monitor. Your monitoring task (the CONDITION "
    "to alert on) is in the system instruction above. This condition is SEMANTIC: it needs "
    "COMPREHENSION and REASONING, not mere object/keyword spotting. You must understand the "
    "user's intent and JUDGE whether what is happening on screen actually MEETS it, then alert "
    "at EACH occurrence where it is satisfied.\n"
    "Each turn, read the stream so far and emit a compact JSON control update. ALWAYS start with "
    "seen, then have_enough_info — looking BEFORE judging, every tick. Whenever have_enough_info "
    "is true, ALSO include event_time_s and answer. You are asked for fps, next_check_s and "
    "question_for_next EVERY tick — always give your current best value, even when it is "
    "unchanged. Fields:\n"
    '  seen              : ALWAYS FIRST — 3-8 words: what is on screen now that is relevant to the condition\n'
    "  have_enough_info  : true when the condition is satisfied on screen NOW; keep reporting true "
    "for as long as it remains satisfied; back to false when it no longer is\n"
    "  event_time_s      : when have_enough_info is true -> the video time in seconds when THIS "
    "occurrence appeared (read it off the 'time Xs' markers in the stream)\n"
    '  answer            : REQUIRED whenever have_enough_info is true -> ONE sentence (UNDER 25 '
    'words) stating WHAT is happening AND WHY it satisfies the condition; else ""\n'
    "  fps               : how densely to sample next (1-3; raise when the scene is busy)\n"
    "  next_check_s      : seconds until the next check (1-3; keep it small — more may come)\n"
    '  question_for_next : a short check to verify on the next turn; else ""\n'
    "YOU DO NOT DECIDE WHEN TO ALERT — the system does. It alerts the user only when "
    "have_enough_info goes false -> true (or when your answer describes a clearly DIFFERENT "
    "occurrence), so you will NEVER double-alert by keeping it true while the same thing stays on "
    "screen. Your only job each tick: judge the level honestly and describe what you see. The "
    "condition typically recurs (about 3 times per video on average), so after it stops, keep "
    "watching for the NEXT occurrence.\n"
    "The 'Already reported' list below shows PAST occurrences WITH THEIR TIMES.\n"
    "Rules: base every judgment on what is actually visible; reason about the user's intent; "
    "NEVER copy the example text.\n"
    "Worked example (a DIFFERENT video — condition: 'alert whenever the video provides specific "
    "logistical details for the match, such as the date, location, or ticket pricing'). The video "
    "is a football club TV commercial: fans in body paint, the crowd roaring, then a poster with "
    "the match date and location, then ticket pricing. Every output starts with seen + "
    "have_enough_info; other fields appear only when they change:\n"
    "At 3s, none of the asked-for details shown yet -> condition NOT satisfied:\n"
    '{"seen":"fans buying green body paint","have_enough_info":false,"fps":1.0,"question_for_next":"Is the date, location or ticket price shown now?"}\n'
    "At 6s still nothing (question unchanged -> omit it):\n"
    '{"seen":"crowd roaring in the stadium","have_enough_info":false}\n'
    "At 16s the poster shows the date and location -> satisfied; describe it and read its time:\n"
    '{"seen":"poster with match date and venue","have_enough_info":true,"event_time_s":16,"answer":"The match date is August 14th and the location is Dairy Farmers Stadium.","question_for_next":"Is the ticket price shown now?"}\n'
    "At 17s the same poster is still on screen -> STILL satisfied; keep reporting it (the system "
    "will not double-alert):\n"
    '{"seen":"same date and venue poster","have_enough_info":true,"event_time_s":16,"answer":"The match date is August 14th and the location is Dairy Farmers Stadium."}\n'
    "At 20s the poster is gone, players celebrating -> no longer satisfied:\n"
    '{"seen":"players celebrating on the pitch","have_enough_info":false}\n'
    "At 23s ticket pricing appears -> satisfied AGAIN by a new detail:\n"
    '{"seen":"ticket prices and purchase info on screen","have_enough_info":true,"event_time_s":23,"answer":"The video now details ticket costs for different groups and gives a purchase website and phone number.","fps":3.0,"next_check_s":3}\n'
    "(all asked-for details reported -> sample sparsely, keep watching in case more appear)\n"
)


# NOTE ON AUDIO: the paper defines ETG "audio-first" (prefer audio triggers); we run the
# audio_dependency=none subset only, so this ICL is purely VISUAL — no audio references.
#
# Paper's task intent: the question names a TRIGGER moment and a TARGET object. When the
# trigger fires (instantaneous, frame-precise), locate the target on a 3x3 grid. Scoring
# is EXACT-MATCH on the grid cell extracted from the answer text (no LLM judge), plus the
# usual ±3s temporal match — so the answer MUST contain exactly one of the nine cell
# names, and firing time must be sharp. Usually ONE trigger per video (max 4).
_EXPLICIT_TARGET_GROUNDING = (
    "\nYou are the CONTROLLER of a live video monitor. Your task (in the system "
    "instruction above) names a TRIGGER moment and a TARGET object: the moment the "
    "trigger happens, report WHERE the target is on screen using EXACTLY ONE of these "
    "nine grid cells (imagine the frame divided 3x3):\n"
    "  top-left | top-center | top-right | center-left | center | center-right | "
    "bottom-left | bottom-center | bottom-right\n"
    "Each turn, read the stream so far and emit a compact JSON control update. ALWAYS "
    "start with seen, then have_enough_info — looking BEFORE judging, every tick. "
    "Whenever have_enough_info is true, ALSO include event_time_s and answer. Include "
    "fps, next_check_s and question_for_next are asked EVERY tick — always give your "
    "current best value, even when unchanged. Fields:\n"
    '  seen              : ALWAYS FIRST — 3-8 words: what is on screen now, relevant to the trigger/target\n'
    "  have_enough_info  : true when the TRIGGER is happening on screen NOW and the "
    "target is visible; back to false once it is over\n"
    "  event_time_s      : when true -> the video time when the trigger occurred (read "
    "the 'time Xs' markers in the stream)\n"
    '  answer            : REQUIRED when true -> ONE sentence UNDER 20 words naming the '
    'trigger AND the target\'s grid cell; it MUST contain exactly one cell name; else ""\n'
    "  fps               : how densely to sample next (1-3; raise when the trigger feels close)\n"
    "  next_check_s      : seconds until the next check (small — the trigger is a precise instant)\n"
    '  question_for_next : a short check to verify on the next turn; else ""\n'
    "YOU DO NOT DECIDE WHEN TO ALERT — the system does (it alerts only when "
    "have_enough_info goes false -> true). The trigger usually happens ONCE; after "
    "reporting, return to false when it is over and keep watching in case it recurs.\n"
    "The 'Already reported' list below shows PAST reports with their times.\n"
    "Rules: report only what is actually visible; the answer must contain exactly ONE "
    "grid-cell name; NEVER copy the example text.\n"
    "Worked example (a DIFFERENT video — a fast-paced montage of short sports clips: "
    "cycling, tennis, cardio, big crowds. Task: 'When the drummer first appears playing "
    "his drums, tell me where his red cap is in the frame.'):\n"
    "At 5s, cyclists racing, no drummer yet:\n"
    '{"seen":"cyclists racing past a crowd","have_enough_info":false,"fps":1.0,"question_for_next":"Has the drummer appeared playing his drums?"}\n'
    "At 12s, tennis rally, still no drummer:\n"
    '{"seen":"tennis player mid-rally","have_enough_info":false}\n'
    "At 19s the drummer appears playing, red cap visible in the upper middle -> TRIGGER; "
    "locate the cap and read the time:\n"
    '{"seen":"drummer playing, red cap upper middle","have_enough_info":true,"event_time_s":19,"answer":"The drummer is now playing - his red cap is in the top-center of the frame."}\n'
    "At 21s the montage cut away, drummer gone -> trigger over:\n"
    '{"seen":"crowd cheering, drummer gone","have_enough_info":false}\n'
)

# Probe-gate arm needs an ETG-specific WRITER prompt too (its generic writer says
# "what+why", which would never contain a grid cell -> auto-wrong on exact match).
_ETG_WRITER_PROMPT = (
    "\nThe trigger just occurred on screen. In ONE sentence UNDER 20 words, state the "
    "trigger and WHERE the target is, using exactly one of: top-left, top-center, "
    "top-right, center-left, center, center-right, bottom-left, bottom-center, "
    "bottom-right. Output only that sentence.")


# NOTE ON AUDIO: the paper defines IEA "audio-first" (prefer whistles/doorbells/
# spoken phrases). We run the audio_dependency=none subset only, so this ICL is
# purely VISUAL — the target is a sharp VISIBLE onset and there are no audio cues.
#
# Task intent: a standing instruction ("Let me know when X happens"). Alert the
# INSTANT the event STARTS on screen; timing is frame-precise (±3s temporal match)
# and the answer is a short natural description (LLM-judged, not exact-match).
# Usually ONE occurrence per video (max ~3), so after it ends keep watching.
_INSTANT_EVENT_ALERT = (
    "\nYou are the CONTROLLER of a live video monitor. Your task (in the system "
    "instruction above) is a STANDING INSTRUCTION to alert the user the moment a "
    "specific EVENT happens on screen (e.g. 'let me know when the audience starts "
    "clapping'). The event is a SHARP ONSET — the instant something STARTS — so your "
    "timing must be precise: fire when it BEGINS, not before. It usually happens ONCE, "
    "but can recur a few times; after it ends, keep watching for the next occurrence.\n"
    "Each turn, read the stream so far and emit a compact JSON control update. ALWAYS "
    "start with seen, then have_enough_info — looking BEFORE judging, every tick. "
    "Whenever have_enough_info is true, ALSO include event_time_s and answer. Include "
    "fps, next_check_s and question_for_next are asked EVERY tick — always give your "
    "current best value, even when unchanged. Fields:\n"
    '  seen              : ALWAYS FIRST — 3-8 words: what is on screen now, relevant to the event\n'
    "  have_enough_info  : true the moment the event is happening on screen NOW; stays true "
    "while it continues; back to false once it is over\n"
    "  event_time_s      : when have_enough_info is true -> the video time in seconds when THIS "
    "occurrence STARTED (read it off the 'time Xs' markers in the stream)\n"
    '  answer            : REQUIRED whenever have_enough_info is true -> ONE natural sentence '
    '(UNDER 20 words) stating WHAT just happened; else ""\n'
    "  fps               : how densely to sample next (1-3; raise when the event feels close)\n"
    "  next_check_s      : seconds until the next check (small — the onset is a precise instant)\n"
    '  question_for_next : a short check to verify on the next turn; else ""\n'
    "YOU DO NOT DECIDE WHEN TO ALERT — the system does. It alerts the user only when "
    "have_enough_info goes false -> true (or when your answer describes a clearly DIFFERENT "
    "occurrence), so you will NEVER double-alert by keeping it true while the same event stays on "
    "screen. Your only job each tick: judge honestly whether the event is happening NOW and "
    "describe what you see.\n"
    "The 'Already reported' list below shows PAST occurrences WITH THEIR TIMES.\n"
    "Rules: base every judgment on what is actually visible; do NOT fire before the event "
    "actually starts; NEVER copy the example text.\n"
    "Worked example (a DIFFERENT video — instruction: 'Let me know when the audience starts "
    "clapping'). The video is a university auditorium: speakers give speeches about the "
    "university, the camera cutting between the podium and the seated audience. Every output "
    "starts with seen + have_enough_info; other fields appear only when they change:\n"
    "At 20s a speaker is talking, the audience is seated and still -> event NOT happening:\n"
    '{"seen":"speaker at podium, audience seated still","have_enough_info":false,"fps":1.0,"question_for_next":"Has the audience started clapping?"}\n'
    "At 34s the speaker is wrapping up his segment, still no clapping (question unchanged -> omit it):\n"
    '{"seen":"speaker finishing his remarks","have_enough_info":false}\n'
    "At 38s the camera cuts to the audience clapping -> event STARTS; describe it and read its time:\n"
    '{"seen":"audience clapping in their seats","have_enough_info":true,"event_time_s":38,"answer":"The audience is clapping now after the opening remarks.","next_check_s":1}\n'
    "At 40s they are still clapping -> STILL happening; keep reporting it (the system will not double-alert):\n"
    '{"seen":"audience still applauding","have_enough_info":true,"event_time_s":38,"answer":"The audience is clapping now after the opening remarks."}\n'
    "At 48s the clapping has stopped and the next speaker is walking up -> event over:\n"
    '{"seen":"applause stopped, new speaker approaching","have_enough_info":false}\n'
    "At 64s the audience breaks into applause again as a family is invited on stage -> a NEW occurrence:\n"
    '{"seen":"audience applauding again","have_enough_info":true,"event_time_s":64,"answer":"The audience has started applauding again as the family is invited to the stage."}\n'
)


# ===========================================================================
# --- FORMAT SECTION (will be replaced by a schema-walk decoder later) ------
# Everything between here and "END FORMAT SECTION" teaches JSON SYNTAX and the
# FIELD LIST only. It contains no task semantics and no worked example, so it
# can be dropped wholesale once code forces the structure tokens.
# ===========================================================================
_FORMAT_HOW = (
    "Each turn, read the stream so far and emit ONE compact JSON control update. "
    "ALWAYS start with seen, then have_enough_info — looking BEFORE judging, every "
    "tick. Whenever have_enough_info is true, ALSO include event_time_s and answer. "
    "fps, next_check_s and question_for_next are asked EVERY tick — always give your "
    "current best value, even when unchanged. Fields:\n"
)

_FORMAT_FIELDS_CORE_A = (
    '  seen              : ALWAYS FIRST — 3-8 words: what is on screen now that matters for your task\n'
)

# have_enough_info is the one field whose MEANING is task-specific; each task's
# semantics block restates it. Here it is described only structurally.
_FORMAT_FIELDS_CORE_B = (
    "  have_enough_info  : a LEVEL, not an edge — true while the thing you are asked "
    "to report is true ON SCREEN NOW; false otherwise\n"
    "  event_time_s      : when have_enough_info is true -> the video time in seconds "
    "this occurrence began; read it off a RECENT 'time Xs' marker in the stream (it "
    "must be within the last 10 seconds)\n"
    '  answer            : REQUIRED whenever have_enough_info is true -> the sentence '
    'the user will actually receive; else ""\n'
    "  fps               : how densely to sample next (1-3; raise when the scene is busy)\n"
    "  next_check_s      : seconds until the next check (0.5-1.5)\n"
    '  question_for_next : a short note-to-self to verify on the next turn; else ""\n'
)

# Both fields have consumers:
# count is fed back into the next prompt as "Your running count"; phase is
# fed back as "Your last recorded state".
_FORMAT_FIELD_COUNT = (
    "  count             : an integer — the running total SO FAR. Carry it "
    "forward on every tick and raise it by exactly 1 when you report a new item\n"
)

_FORMAT_FIELD_PHASE = (
    "  phase             : 1-4 plain words naming the state the video is in RIGHT NOW. "
    "Carry it forward on every tick; change it only when the state really changes\n"
)

_FORMAT_GATE = (
    "YOU DO NOT DECIDE WHEN TO ALERT — the system does. It alerts the user only when "
    "have_enough_info goes false -> true (or when your answer describes a clearly "
    "DIFFERENT occurrence), so you will NEVER double-alert by keeping it true while the "
    "same thing stays on screen. Your only job each tick: judge the level honestly and "
    "describe what you see.\n"
    "The 'Already reported' list below shows PAST reports WITH THEIR TIMES. It is your "
    "ONLY memory of what you have already told the user — read it before every "
    "judgment.\n"
)

_FORMAT_BLOCK_BASE = _FORMAT_HOW + _FORMAT_FIELDS_CORE_A + _FORMAT_FIELDS_CORE_B + _FORMAT_GATE
_FORMAT_BLOCK_COUNT = (_FORMAT_HOW + _FORMAT_FIELDS_CORE_A + _FORMAT_FIELDS_CORE_B
                       + _FORMAT_FIELD_COUNT + _FORMAT_GATE)
_FORMAT_BLOCK_STATE = (_FORMAT_HOW + _FORMAT_FIELDS_CORE_A + _FORMAT_FIELDS_CORE_B
                       + _FORMAT_FIELD_PHASE + _FORMAT_GATE)
# ===========================================================================
# --- END FORMAT SECTION ---------------------------------------------------
# ===========================================================================


# ---------------------------------------------------------------------------
# 1. snapshot_counting
# ---------------------------------------------------------------------------
# NOTE ON AUDIO: the paper allows audio triggers for this task; we run the
# audio_dependency=none subset only and ingest no audio, so this ICL is purely
# VISUAL — the trigger is something that becomes VISIBLE on screen.
#
# Task intent: the question names a TRIGGER moment and a class of THINGS. At the
# instant the trigger happens, count how many of those things are visible AT THAT
# MOMENT (a snapshot, not a running tally). There is exactly ONE trigger per video.
# Scoring: ±3s temporal match AND the first integer in the answer must equal the
# ground-truth count -> the answer MUST begin with the number as a digit.
_ROLE_SNAPSHOT_COUNTING = (
    "\nYou are the CONTROLLER of a live video monitor. Your task (in the system "
    "instruction above) names a TRIGGER moment and a kind of THING to count: the "
    "moment the trigger happens on screen, count how many of those things are "
    "visible AT THAT INSTANT and report the number. This is a SNAPSHOT count — you "
    "count what is in the frame at the trigger moment, you do NOT add up things seen "
    "earlier in the video. The trigger happens ONCE per video, so there is normally "
    "exactly one report to make: get its timing and its number right.\n"
)

_SEMANTICS_SNAPSHOT_COUNTING = (
    "HOW YOU TRACK THE COUNT: set `count` to the number of items you can see when "
    "the trigger is on screen; before the trigger, keep count at 0. Your prompt "
    "shows your count from the previous tick, so you do not need to re-derive it.\n"
    "What have_enough_info means here: true while the trigger condition is visibly "
    "true on screen AND you can see the things to be counted; false before it and "
    "after it is gone. Before the trigger, there is nothing to report yet — keep looking.\n"
    "HOW TO WRITE THE ANSWER — this is scored by pulling the FIRST NUMBER out of your "
    "sentence:\n"
    "  * The answer MUST BEGIN with the count written as a DIGIT: '5 prices are ...'\n"
    "  * NEVER let any other number come first — no years, model numbers, prices, "
    "times or scores before the count. Write '3 balls are in the net' and never "
    "'the 2018 model shows 3 balls'.\n"
    "  * Then ONE short clause (under 20 words total) naming what you counted and the "
    "trigger, so the user knows what the number refers to.\n"
    "Rules: count only what you can actually SEE in the frame right now — do not guess "
    "or round; if things are partly hidden, count the ones you can distinguish; "
    "re-count on each tick while the trigger is on screen and keep the same number if "
    "nothing changed; NEVER copy the example text.\n"
    "Worked example (a DIFFERENT video — task: 'When the baker takes the tray out of "
    "the oven, count how many croissants are on the tray.'). The video is a small "
    "bakery vlog: dough being mixed, shaped, proved, baked, then packed into boxes. "
    "Every output starts with seen + have_enough_info; other fields appear only when "
    "they change:\n"
    "At 4s dough is being mixed — the trigger has not happened, nothing to count yet:\n"
    '{"seen":"baker mixing dough in a bowl","have_enough_info":false,"fps":1.0,"question_for_next":"Has the tray come out of the oven yet?","count":0}\n'
    "At 21s he is shaping the pastries — still no trigger (question unchanged -> omit it):\n"
    '{"seen":"hands rolling croissant shapes","have_enough_info":false,"count":0}\n'
    "At 33s the tray goes INTO the oven — close, so look more often:\n"
    '{"seen":"tray slid into the hot oven","have_enough_info":false,"fps":3.0,"next_check_s":0.5,"count":0}\n'
    "At 46s he pulls the tray out; the croissants are clearly separated on it -> "
    "TRIGGER; count them and read the time off the marker:\n"
    '{"seen":"baker lifting tray out of oven","have_enough_info":true,"event_time_s":46,"answer":"8 croissants are on the tray the baker has just taken out of the oven.","count":8}\n'
    "At 48s the same tray is still in his hands -> STILL the same trigger; re-count, "
    "keep the same number and the same event_time_s (the system will not double-alert):\n"
    '{"seen":"same tray still held up","have_enough_info":true,"event_time_s":46,"answer":"8 croissants are on the tray the baker has just taken out of the oven.","count":8}\n'
    "At 55s the tray is on the counter and the camera has moved to the packing boxes "
    "-> the trigger moment is over:\n"
    '{"seen":"empty counter and packing boxes","have_enough_info":false,"fps":1.0,"next_check_s":1.5,"count":8}\n'
)

_SNAPSHOT_COUNTING = (_ROLE_SNAPSHOT_COUNTING + _FORMAT_BLOCK_COUNT
                      + _SEMANTICS_SNAPSHOT_COUNTING)


# ---------------------------------------------------------------------------
# 2. cumulative_counting
# ---------------------------------------------------------------------------
# NOTE ON AUDIO: the paper allows audio triggers for this task; we run the
# audio_dependency=none subset only and ingest no audio, so this ICL is purely
# VISUAL — every occurrence must be SEEN.
#
# Task intent: count REPETITIONS of one kind of action over the whole video ("count
# how many times X happens"). One report per occurrence, each carrying the running
# total at that moment. Scoring: ±3s temporal match AND the first integer in the
# answer must equal the ground-truth running index -> digit-initial answers, and the
# count is re-derived from the "Already reported" list (the only real memory).
_ROLE_CUMULATIVE_COUNTING = (
    "\nYou are the CONTROLLER of a live video monitor. Your task (in the system "
    "instruction above) asks you to COUNT HOW MANY TIMES one particular thing happens "
    "across the video. The SAME action repeats — you must report EVERY occurrence, "
    "once each, as it happens, and each report carries the RUNNING TOTAL up to that "
    "moment. Occurrences may be seconds apart or minutes apart; there are typically "
    "2-6 of them.\n"
)

_SEMANTICS_CUMULATIVE_COUNTING = (
    "HOW YOU REMEMBER THE COUNT: your prompt shows your running count from the "
    "previous tick. Carry it forward in the `count` field every tick: if a new "
    "occurrence is happening now (have_enough_info true on a new event), set count "
    "to the previous value plus one; otherwise keep it the same. The 'Already "
    "reported' list is your backup check. Also keep the total in question_for_next, "
    "e.g. 'total so far 2; watch for a 3rd pour'.\n"
    "What have_enough_info means here: true FOR AS LONG AS an occurrence is visibly "
    "happening on screen right now — the whole time, not just its first instant. False "
    "whenever no occurrence is on screen. Do not try to drop back to false in order to "
    "let the next one be reported: the system watches for the false->true transition "
    "itself and will never double-report. Just describe the level honestly.\n"
    "HOW TO WRITE THE ANSWER — this is scored by pulling the FIRST NUMBER out of your "
    "sentence:\n"
    "  * The answer MUST BEGIN with the running total written as a DIGIT: "
    "'3 pours so far — ...'\n"
    "  * NEVER let any other number come first (no times, ages, prices or scores "
    "before it), and write the digit, not the word ('3', never 'three').\n"
    "  * Then ONE short clause (under 20 words total) saying what just happened, so "
    "the user can tell the occurrences apart.\n"
    "Rules: count an occurrence only when you actually SEE it complete — do not count "
    "the same one twice because it is still on screen, and do not count something you "
    "merely expect; if you are unsure whether an occurrence finished, DEFER with a "
    "question and a short next_check_s; NEVER copy the example text.\n"
    "Worked example (a DIFFERENT video — task: 'Count how many times the barista pours "
    "steamed milk into a cup.'). The video is a coffee-shop day-in-the-life: grinding, "
    "tamping, extracting shots, pouring, wiping down, customers at the till. Every "
    "output starts with seen + have_enough_info; other fields appear only when they "
    "change:\n"
    "At 6s the grinder is running — nothing counted yet:\n"
    '{"seen":"barista grinding coffee beans","have_enough_info":false,"fps":1.0,"question_for_next":"total so far 0; has a milk pour started?","count":0}\n'
    "At 18s a shot is extracting into a cup — still not a pour:\n"
    '{"seen":"espresso extracting into a cup","have_enough_info":false,"count":0}\n'
    "At 24s he tips the jug and pours milk into the cup -> FIRST occurrence (nothing "
    "in 'Already reported' yet, so the total is 1):\n"
    '{"seen":"milk poured from jug into cup","have_enough_info":true,"event_time_s":24,"answer":"1 pour so far — the barista pours steamed milk into the first cup.","question_for_next":"total so far 1; watch for a 2nd pour","count":1}\n'
    "At 25s he is still pouring the SAME cup -> still the same occurrence; keep the "
    "same total, time and answer (the system will not double-alert):\n"
    '{"seen":"still pouring the same cup","have_enough_info":true,"event_time_s":24,"answer":"1 pour so far — the barista pours steamed milk into the first cup.","count":1}\n'
    "At 27s the cup is on the saucer, pour finished -> drop to false so the next pour "
    "can be reported:\n"
    '{"seen":"finished cup placed on saucer","have_enough_info":false,"count":1}\n'
    "At 41s he pours milk into a second cup -> NEW occurrence; 'Already reported' has "
    "1 line, so the total is 2:\n"
    '{"seen":"milk poured into a second cup","have_enough_info":true,"event_time_s":41,"answer":"2 pours so far — he pours milk into a second cup for the next order.","question_for_next":"total so far 2; watch for a 3rd pour","count":2}\n'
    "At 44s he is wiping the steam wand -> occurrence over:\n"
    '{"seen":"barista wiping the steam wand","have_enough_info":false,"count":2}\n'
    "At 70s a third pour begins -> report it with the total 3:\n"
    '{"seen":"milk poured into a third cup","have_enough_info":true,"event_time_s":70,"answer":"3 pours so far — he pours milk into a third cup for a waiting customer.","question_for_next":"total so far 3; watch for a 4th pour","count":3}\n'
)

_CUMULATIVE_COUNTING = (_ROLE_CUMULATIVE_COUNTING + _FORMAT_BLOCK_COUNT
                        + _SEMANTICS_CUMULATIVE_COUNTING)


# ---------------------------------------------------------------------------
# 3. dedup_counting
# ---------------------------------------------------------------------------
# NOTE ON AUDIO: the paper allows audio triggers for this task; we run the
# audio_dependency=none subset only and ingest no audio, so this ICL is purely
# VISUAL — identity is judged from appearance alone.
#
# Task intent: count how many DIFFERENT things (people, characters, settings,
# environments) are featured across the video. Only the FIRST appearance of each
# distinct thing counts; a thing coming back a second time must NOT be counted or
# reported again. Scoring: ±3s temporal match AND the first integer in the answer
# must equal the ground-truth running index of distinct items.
_ROLE_DEDUP_COUNTING = (
    "\nYou are the CONTROLLER of a live video monitor. Your task (in the system "
    "instruction above) asks how many DIFFERENT things (people, characters, settings, "
    "backgrounds...) appear across the video. The rule that defines this task: each "
    "distinct thing counts EXACTLY ONCE, at its FIRST appearance. When something you "
    "have already counted comes back on screen, you must NOT count or report it again. "
    "So every tick you are answering one question: is this a thing I have never "
    "reported before?\n"
)

_SEMANTICS_DEDUP_COUNTING = (
    "HOW YOU REMEMBER WHAT IS ALREADY COUNTED: your prompt shows your running count "
    "from the previous tick. That list IS your inventory: each line names one thing "
    "you have already counted. Before reporting, check the thing on screen against "
    "every line in 'Already reported'. If it matches one, it is a REPEAT -> "
    "have_enough_info stays false and count stays the same. If it matches none, "
    "it is NEW -> set count to the previous value plus one. Keep it in "
    "question_for_next too, e.g. 'counted so far: beagle, labrador (2)'.\n"
    "What have_enough_info means here: true FOR AS LONG AS something you have NEVER "
    "counted is clearly featured on screen. Stay true the whole time it is featured. "
    "False when it is gone, or when everything on screen is already in your list. Do "
    "not try to drop back to false to let the next new thing be reported: the system "
    "watches for the false->true transition itself and will never double-report.\n"
    "HOW TO WRITE THE ANSWER — this is scored by pulling the FIRST NUMBER out of your "
    "sentence:\n"
    "  * The answer MUST BEGIN with the running total of DISTINCT things, written as a "
    "DIGIT: '2 different dogs so far — ...'\n"
    "  * NEVER let any other number come first, and write the digit, not the word.\n"
    "  * Then ONE short clause (under 20 words total) DESCRIBING the new thing "
    "distinctively — colour, clothing, role, location — because that description is "
    "what lets you recognise a repeat later.\n"
    "Rules: judge identity from what you can SEE; a different camera angle on the same "
    "thing is NOT a new thing; two things that merely look similar but are clearly "
    "different (different colour, different person) ARE new; when unsure whether it is "
    "a repeat, DEFER with a question rather than counting it; NEVER copy the example "
    "text.\n"
    "Worked example (a DIFFERENT video — task: 'Count how many different dogs are "
    "featured alone in this animal-shelter video.'). The video walks down a kennel "
    "aisle, cutting to close-ups of individual dogs and back to the aisle, and some "
    "dogs are shown more than once. Every output starts with seen + have_enough_info; "
    "other fields appear only when they change:\n"
    "At 5s a caretaker walks down the aisle, no single dog featured:\n"
    '{"seen":"caretaker walking down kennel aisle","have_enough_info":false,"fps":1.0,"question_for_next":"counted so far: none (0)","count":0}\n'
    "At 9s the shot cuts to a brown beagle alone in the frame -> NEW (nothing in "
    "'Already reported'), so the total is 1:\n"
    '{"seen":"brown beagle alone in close-up","have_enough_info":true,"event_time_s":9,"answer":"1 different dog so far — a brown beagle with long ears featured alone.","question_for_next":"counted so far: brown beagle (1)","count":1}\n'
    "At 11s the same beagle is still in frame from a lower angle -> the SAME first "
    "appearance; keep the same total, time and answer (the system will not "
    "double-alert):\n"
    '{"seen":"same beagle, lower camera angle","have_enough_info":true,"event_time_s":9,"answer":"1 different dog so far — a brown beagle with long ears featured alone.","count":1}\n'
    "At 14s back to the aisle, no dog featured -> false:\n"
    '{"seen":"empty aisle between the kennels","have_enough_info":false,"count":1}\n'
    "At 22s a black labrador is featured alone -> not in the list -> NEW, total 2:\n"
    '{"seen":"black labrador alone in close-up","have_enough_info":true,"event_time_s":22,"answer":"2 different dogs so far — a black labrador lying alone against the kennel door.","question_for_next":"counted so far: brown beagle, black labrador (2)","count":2}\n'
    "At 35s the BROWN BEAGLE is on screen again -> it is already on the list -> this "
    "is a REPEAT; do NOT count it and do NOT report it:\n"
    '{"seen":"the brown beagle again, already counted","have_enough_info":false,"count":2}\n'
    "At 48s a white terrier is featured alone -> NEW, total 3:\n"
    '{"seen":"white terrier alone on a blanket","have_enough_info":true,"event_time_s":48,"answer":"3 different dogs so far — a small white terrier sitting alone on a blanket.","question_for_next":"counted so far: beagle, labrador, terrier (3)","count":3}\n'
)

_DEDUP_COUNTING = (_ROLE_DEDUP_COUNTING + _FORMAT_BLOCK_COUNT
                   + _SEMANTICS_DEDUP_COUNTING)


# ---------------------------------------------------------------------------
# 4. realtime_state_monitor
# ---------------------------------------------------------------------------
# NOTE ON AUDIO: the paper allows audio triggers for this task; we run the
# audio_dependency=none subset only and ingest no audio, so this ICL is purely
# VISUAL — state changes must be visible on screen.
#
# Task intent: the question names a STATE VARIABLE to track (location, kind of
# footage, activity, ...). Report every TRANSITION of that variable. Scoring: ±3s
# temporal match AND the answer, lowercased and unquoted, must equal `state_to`
# EXACTLY (upstream `utils/online_parser.py` takes the WHOLE payload as the state;
# see metrics._extract_state) -> the answer is a BARE STATE LABEL, never a sentence.
#
# A sentence answer ("The setting switched from X to Y.") scores content_acc = 0.000
# under the upstream scorer (0/139 matched emits in a full development run) — a
# sentence can never equal a bare state name. The
# vocabulary is NOT handed to the model (benchmark `states` is ground-truth metadata
# and dataset.py deliberately does not load it), so the only legitimate source for the
# exact wording is the task question itself, which enumerates the states in 22% of
# samples. Hence the two rules below: bare label, and copy the question's wording.
_ROLE_REALTIME_STATE_MONITOR = (
    "\nYou are the CONTROLLER of a live video monitor. Your task (in the system "
    "instruction above) names ONE thing whose STATE you must track for the whole video "
    "— where the subject is, what kind of footage is being shown, what activity is "
    "under way. You report nothing while the state holds; you report the moment it "
    "CHANGES, and what you report is the NAME OF THE NEW STATE — nothing else. There "
    "are typically 3-6 changes in a video.\n"
)

# `phase` is wired end-to-end: the model fills `phase` every tick, controller.py
# persists it in `state`, and feeds it back as "Your last recorded state (phase): ..."
# in the next tick's prompt. have_enough_info is expressed as comparing what is on
# screen against `phase`.
_SEMANTICS_REALTIME_STATE_MONITOR = (
    "HOW YOU REMEMBER THE CURRENT STATE: put it in `phase` on EVERY tick, "
    "including quiet ticks — that is what makes the next change detectable. Your "
    "prompt shows your last recorded phase; carry it forward unchanged when the "
    "state holds, and update it when you report a change. The 'Already reported' "
    "list is your durable backup.\n"
    "What have_enough_info means here: true FOR AS LONG AS the state you can SEE on "
    "screen differs from your current `phase`. It becomes false again naturally, "
    "once you update phase to match what is on screen and the two agree. Do "
    "not try to force it back to false: the system watches for the false->true "
    "transition itself and will never double-report the same change.\n"
    "HOW TO WRITE THE ANSWER — this task is graded by EXACT STRING MATCH against the "
    "name of the NEW state, so the answer is NOT a sentence:\n"
    "  * The answer is ONLY THE NAME OF THE NEW STATE, 1-4 words. No verb, no 'The "
    "setting switched from ... to ...', no mention of the OLD state, no leading 'a' / "
    "'an' / 'the', no full stop, no quotes. Just the label: forest trail | sitting | "
    "kitchen | Question 3 | wheel close-up.\n"
    "  * IF THE TASK SENTENCE LISTS THE POSSIBLE STATES ('...switches between walking, "
    "standing, or sitting'), the answer MUST be exactly one of those listed states, "
    "COPIED WORD-FOR-WORD out of the task. Do not re-word, re-order or pluralise it: "
    "if the task says 'indoor', answering 'indoors' is WRONG.\n"
    "  * If the task does NOT list them, use the shortest literal label the video "
    "itself would use — 'kitchen', not 'a brightly lit kitchen with white cabinets'. "
    "Reuse the SAME label every time that state comes back.\n"
        "  * Do not add commentary, opinions or guesses about what comes next.\n"
    "Rules: a camera cut back to something you already showed is still a CHANGE if the "
    "state really became different; a slow pan within the same place is NOT a change; "
    "a brief flash or insert that immediately reverts is NOT a change — if unsure, "
    "DEFER with a question and a short next_check_s; NEVER copy the example text.\n"
    "Worked example (a DIFFERENT video — task: 'Monitor the scene's location and let "
    "me know whenever the setting changes.'). The video is a day-hike vlog: it starts "
    "in town, goes up a forest trail, reaches a lake, then ends at a campsite. Every "
    "output starts with seen + have_enough_info; other fields appear only when they "
    "change:\n"
    "At 4s the hiker is walking past shopfronts -> establish the starting state, "
    "nothing to report:\n"
    '{"seen":"hiker walking past town shopfronts","have_enough_info":false,"fps":1.0,"question_for_next":"has the setting changed?","phase":"city street"}\n'
    "At 12s still on the street (state unchanged, question unchanged -> stay short):\n"
    '{"seen":"same street, crossing at a light","have_enough_info":false,"phase":"city street"}\n'
    "At 19s the shot is now a narrow path under trees -> the state CHANGED; the answer "
    "is the NEW state's bare name, not a sentence about the change:\n"
    '{"seen":"narrow path under tall trees","have_enough_info":true,"event_time_s":19,"answer":"forest trail","question_for_next":"has the setting changed?","phase":"forest trail"}\n'
    "At 21s still the same trail, the change already reported -> keep it true with the "
    "SAME answer (the system will not double-alert):\n"
    '{"seen":"still walking the forest trail","have_enough_info":true,"event_time_s":19,"answer":"forest trail","phase":"forest trail"}\n'
    "At 30s the trail is clearly the settled state -> drop to false so the NEXT change "
    "can fire:\n"
    '{"seen":"trail continues through the trees","have_enough_info":false,"phase":"forest trail"}\n'
    "At 52s the trees open onto stones and water -> a NEW change:\n"
    '{"seen":"open water and stony shoreline","have_enough_info":true,"event_time_s":52,"answer":"rocky lake shore","question_for_next":"has the setting changed?","phase":"rocky lake shore"}\n'
    "At 58s a two-second insert of the map on his phone, then straight back to the "
    "shore -> a flash, NOT a state change:\n"
    '{"seen":"brief map insert, back to shore","have_enough_info":false,"phase":"rocky lake shore"}\n'
    "Second worked example, ONE tick only, to show the copy-the-task-wording rule "
    "(task: 'Monitor Jeffrey's posture and update me when he switches between walking, "
    "standing, or sitting.'). He was standing; at 88s he lowers himself into a chair. "
    "The task lists the states, so the answer is that list's word, verbatim — 'sitting', "
    "NOT 'sitting down', 'seated' or 'Jeffrey switched from standing to sitting.':\n"
    '{"seen":"Jeffrey lowering himself into a chair","have_enough_info":true,"event_time_s":88,"answer":"sitting","question_for_next":"has his posture changed?","phase":"sitting"}\n'
)

_REALTIME_STATE_MONITOR = (_ROLE_REALTIME_STATE_MONITOR + _FORMAT_BLOCK_STATE
                           + _SEMANTICS_REALTIME_STATE_MONITOR)

# Same argument as _ETG_WRITER_PROMPT: the generic writer says "what happened and
# why", which is a sentence and therefore auto-wrong under exact match on the state
# name. The probe-gate arm needs its own.
_RSM_WRITER_PROMPT = (
    "\nThe tracked state just changed on screen. Output ONLY the NAME OF THE NEW "
    "STATE — 1-4 words, no verb, no article, no full stop, no mention of the old "
    "state. If the task sentence lists the possible states, copy the matching one "
    "WORD-FOR-WORD from that list.")


# ---------------------------------------------------------------------------
# 5. event_narration
# ---------------------------------------------------------------------------
# NOTE ON AUDIO: the paper allows audio triggers for this task; we run the
# audio_dependency=none subset only and ingest no audio, so this ICL is purely
# VISUAL — narrate only what is visible in the frames.
#
# Task intent: a standing request for a RUNNING COMMENTARY. The model emits a fresh
# description each time the video moves into a new stage of what it is showing.
# Ground truth has 4-6 triggers per video, roughly 20-40s apart, each a 1-3 sentence
# concrete description. Scoring: ±3s temporal match AND an LLM judge comparing our
# sentence to the ground-truth description (score >= 3 counts) -> content must be
# specific: subjects, actions, setting, one concrete visual detail.
_ROLE_EVENT_NARRATION = (
    "\nYou are the CONTROLLER of a live video monitor, acting as a COMMENTATOR. Your "
    "task (in the system instruction above) asks you to narrate what is happening as "
    "the video unfolds. You do not wait for one special moment — you speak each time "
    "the video moves into a NEW STAGE of the thing it is showing, and you stay quiet "
    "in between. A video normally has 4-6 such stages, roughly 20-40 seconds apart.\n"
)

_SEMANTICS_EVENT_NARRATION = (
    "What counts as a NEW STAGE: the subjects, the activity, the place or the phase of "
    "the process has clearly MOVED ON from what you last reported. More of the same "
    "thing — the same action continuing, a new camera angle on the same action, a cut "
    "back to something you already described — is NOT a new stage. Check the 'Already "
    "reported' list below before every judgment: if your next sentence would say "
    "roughly what one of those lines already says, the answer is no.\n"
    "What have_enough_info means here: true FOR AS LONG AS what is on screen is a stage "
    "you have NOT yet described in the 'already told the user' list. It becomes false "
    "naturally once you have described it and nothing newer has begun. Do not try to "
    "force it back to false: the system watches for the false->true transition itself "
    "and will never double-report the same stage.\n"
    "HOW TO WRITE THE ANSWER — a human reader is comparing it against a written "
    "description of the same moment, so BE SPECIFIC:\n"
    "  * ONE or TWO sentences, 20-40 words. This is longer than an alert — vagueness "
    "is the main way this task fails.\n"
    "  * Name the SUBJECTS (who or what), the ACTION (what they are doing), and at "
    "least one CONCRETE VISUAL DETAIL you can actually see: a colour, a piece of "
    "equipment, the setting, a number, or text visible on screen.\n"
    "  * Write it in the present tense, as commentary to the user. Do not editorialise "
    "and do not speculate about what will happen next.\n"
    "  * Never write filler like 'the video continues' or 'something is happening'.\n"
    "Rules: describe only what is actually visible in the recent frames; if a stage is "
    "still forming, DEFER with a question and a short next_check_s rather than "
    "narrating a guess; NEVER copy the example text.\n"
    "Worked example (a DIFFERENT video — task: 'As you watch, narrate the key stages "
    "of this rooftop solar-panel installation.'). The video follows a crew from "
    "arrival to switch-on. Every output starts with seen + have_enough_info; other "
    "fields appear only when they change:\n"
    "At 6s the van is parked and crates are being unloaded — the work has not started:\n"
    '{"seen":"crew unloading crates from a van","have_enough_info":false,"fps":1.0,"question_for_next":"Has work started up on the roof?"}\n'
    "At 12s two installers carry panels up a ladder -> the FIRST stage has begun:\n"
    '{"seen":"installers carrying panels up ladder","have_enough_info":true,"event_time_s":12,"answer":"Work begins on the roof: two installers in high-visibility vests carry black solar panels up an extension ladder while a third steadies its base on the driveway."}\n'
    "At 15s they are still on the ladder with the next panel -> SAME stage; keep it "
    "true with the same answer (the system will not double-alert):\n"
    '{"seen":"second panel going up the ladder","have_enough_info":true,"event_time_s":12,"answer":"Work begins on the roof: two installers in high-visibility vests carry black solar panels up an extension ladder while a third steadies its base on the driveway."}\n'
    "At 22s still ferrying panels — nothing new to say -> drop to false:\n"
    '{"seen":"more panels stacked on the roof","have_enough_info":false,"question_for_next":"Have they started fixing anything to the roof?"}\n'
    "At 38s they are drilling and bolting rails to the tiles -> a NEW stage:\n"
    '{"seen":"drilling aluminium rails onto tiles","have_enough_info":true,"event_time_s":38,"answer":"The crew is now bolting long aluminium mounting rails to the tiles, drilling anchor points along two rows and checking each rail with a spirit level."}\n'
    "At 64s panels are being clamped onto those rails, half the roof covered -> the "
    "next stage:\n"
    '{"seen":"panels clamped onto the rails","have_enough_info":true,"event_time_s":64,"answer":"Panels are going down onto the rails now, clamped in place one by one; roughly half the south-facing roof is already covered in dark glass."}\n'
    "At 92s the wiring is run into an inverter on the garage wall -> the final stage:\n"
    '{"seen":"cables wired into a wall inverter","have_enough_info":true,"event_time_s":92,"answer":"Downstairs an electrician wires the array into a white inverter on the garage wall and closes the isolator switch; the display lights up with the first output reading."}\n'
)

_EVENT_NARRATION = (_ROLE_EVENT_NARRATION + _FORMAT_BLOCK_BASE
                    + _SEMANTICS_EVENT_NARRATION)


# ---------------------------------------------------------------------------
# 6. sequential_step_instruction
# ---------------------------------------------------------------------------
# NOTE ON AUDIO: the paper allows audio triggers for this task; we run the
# audio_dependency=none subset only and ingest no audio, so this ICL is purely
# VISUAL — each step is recognised from what is visible on screen.
#
# Task intent: the user is FOLLOWING ALONG with a procedure and wants to be told what
# to DO at each step, at the moment that step comes up on screen. Ground truth is
# written in the IMPERATIVE, addressed to the user ("Spray the leather cleaner onto a
# sponge and begin scrubbing the seat bolster."), NOT as a description of what the
# person in the video did. Scoring: ±3s temporal match AND an LLM judge against that
# instruction -> imperative mood, with the concrete parameters visible on screen.
_ROLE_SEQUENTIAL_STEP_INSTRUCTION = (
    "\nYou are the CONTROLLER of a live video monitor, acting as a HANDS-FREE GUIDE. "
    "The user is doing the procedure in the system instruction above THEMSELVES, "
    "alongside the video, and wants you to tell them what to do at each step, exactly "
    "when that step comes up. So you are not describing the video — you are "
    "INSTRUCTING the user. A procedure normally has 4-7 steps.\n"
)

_SEMANTICS_SEQUENTIAL_STEP_INSTRUCTION = (
    "What counts as a NEW STEP: the person on screen starts a distinct new action of "
    "the procedure — a new tool, a new material, a new part being worked on. Repeating "
    "or continuing the SAME action is not a new step. Your memory of which steps you "
    "have already given is the 'Already reported' list below: read it first, count the "
    "lines, and the step you are about to give is the next number. Keep a reminder in "
    "question_for_next, e.g. 'given steps 1-2; what is step 3?'.\n"
    "What have_enough_info means here: true FOR AS LONG AS a step is underway that you "
    "have NOT yet given the instruction for — so the user is told as the step starts "
    "and can follow along, not after it is finished. It becomes false naturally once "
    "you have given that instruction and no later step has begun. Do not try to force "
    "it back to false: the system watches for the false->true transition itself and "
    "will never double-report the same step.\n"
    "HOW TO WRITE THE ANSWER — this is the part most likely to go wrong:\n"
    "  * Write an INSTRUCTION TO THE USER in the imperative: 'Loosen the outer roots "
    "with your fingers...'. Do NOT narrate the video ('the person loosens the roots' "
    "is WRONG for this task).\n"
    "  * Start with 'Step N — ' using the number you worked out from the reported "
    "list, then ONE or TWO sentences, 15-35 words.\n"
    "  * Include the CONCRETE PARAMETERS you can see: the tool, the quantity, the "
    "setting, the direction, the part being worked on. 'Add about a third of a pot of "
    "fresh mix' beats 'add some soil'.\n"
    "  * One step per answer. Do not stack several steps into one sentence and do not "
    "give a step the video has not reached.\n"
    "Rules: instruct only what you can actually SEE being done; if a step is ambiguous, "
    "DEFER with a question and a short next_check_s rather than inventing it; NEVER "
    "copy the example text.\n"
    "Worked example (a DIFFERENT video — task: 'I'm repotting a houseplant along with "
    "this tutorial. Guide me through each step as it happens.'). The video shows a "
    "plant being moved into a bigger pot on a kitchen table. Every output starts with "
    "seen + have_enough_info; other fields appear only when they change:\n"
    "At 3s pots, a bag of compost and scissors are laid out — nothing has started:\n"
    '{"seen":"pots, compost and scissors on table","have_enough_info":false,"fps":1.0,"question_for_next":"given steps: none; has step 1 started?"}\n'
    "At 7s she tips the plant sideways and eases it out of its old pot -> STEP 1 "
    "begins:\n"
    '{"seen":"plant being eased out of pot","have_enough_info":true,"event_time_s":7,"answer":"Step 1 — Tip the pot on its side, hold the base of the stem, and ease the plant out, supporting the root ball as it slides free.","question_for_next":"given step 1; what is step 2?"}\n'
    "At 9s she is still working it loose -> SAME step; keep it true with the same "
    "answer (the system will not double-alert):\n"
    '{"seen":"still working the root ball loose","have_enough_info":true,"event_time_s":7,"answer":"Step 1 — Tip the pot on its side, hold the base of the stem, and ease the plant out, supporting the root ball as it slides free."}\n'
    "At 14s the root ball is on the table and nothing new has started -> drop to "
    "false:\n"
    '{"seen":"root ball resting on the table","have_enough_info":false}\n'
    "At 21s she is teasing the outer roots apart and snipping some -> STEP 2:\n"
    '{"seen":"fingers teasing roots, scissors ready","have_enough_info":true,"event_time_s":21,"answer":"Step 2 — Loosen the tangled outer roots with your fingers, then trim any dark or mushy ones away with clean scissors.","question_for_next":"given steps 1-2; what is step 3?"}\n'
    "At 33s she pours compost into the new, larger pot -> STEP 3:\n"
    '{"seen":"compost poured into larger pot","have_enough_info":true,"event_time_s":33,"answer":"Step 3 — Part-fill the new, slightly larger pot with fresh potting mix, about a third of the way up, and firm it down lightly.","question_for_next":"given steps 1-3; what is step 4?"}\n'
    "At 46s the plant is set in and mix is packed round the sides -> STEP 4:\n"
    '{"seen":"plant centred, mix packed around","have_enough_info":true,"event_time_s":46,"answer":"Step 4 — Centre the plant in the new pot and backfill around it with mix, pressing gently until the root ball sits just below the rim.","question_for_next":"given steps 1-4; what is step 5?"}\n'
)

_SEQUENTIAL_STEP_INSTRUCTION = (_ROLE_SEQUENTIAL_STEP_INSTRUCTION + _FORMAT_BLOCK_BASE
                                + _SEMANTICS_SEQUENTIAL_STEP_INSTRUCTION)


# ---------------------------------------------------------------------------
# The complete mapping. Drop-in replacement for
# AsyncOmniConfig.task_controller_prompts:
#     cfg = dataclasses.replace(cfg, task_controller_prompts=TASK_CONTROLLER_PROMPTS)
# Keys are exactly evaluation.dataset.ALL_TASKS.
# ---------------------------------------------------------------------------
TASK_CONTROLLER_PROMPTS: dict[str, str] = {
    # --- the 3 originally hand-written in config.py ---
    "semantic_condition_alert": _SEMANTIC_CONDITION_ALERT,
    "explicit_target_grounding": _EXPLICIT_TARGET_GROUNDING,
    "instant_event_alert": _INSTANT_EVENT_ALERT,
    # --- new in this file ---
    "snapshot_counting": _SNAPSHOT_COUNTING,
    "cumulative_counting": _CUMULATIVE_COUNTING,
    "dedup_counting": _DEDUP_COUNTING,
    "realtime_state_monitor": _REALTIME_STATE_MONITOR,
    "event_narration": _EVENT_NARRATION,
    "sequential_step_instruction": _SEQUENTIAL_STEP_INSTRUCTION,
}

# The 6 tasks added later (the other 3 were the original hand-written set).
NEW_TASKS: tuple[str, ...] = (
    "snapshot_counting", "cumulative_counting", "dedup_counting",
    "realtime_state_monitor", "event_narration", "sequential_step_instruction",
)

# Extra persistent DSL fields each new task's prompt teaches, beyond the shared
# seen / have_enough_info / event_time_s / answer / fps / next_check_s /
# question_for_next. count and phase are wired end-to-end (consumed by controller.py).
# There is no free-form `note` field (it duplicated seen + question_for_next).
TASK_STATE_FIELDS: dict[str, tuple[str, ...]] = {
    "snapshot_counting": ("count",),
    "cumulative_counting": ("count",),
    "dedup_counting": ("count",),
    "realtime_state_monitor": ("phase",),
    "event_narration": (),
    "sequential_step_instruction": (),
}

# (role, format, semantics) per new task, so the FORMAT part can be swapped or
# stripped mechanically when the schema-walked decoder lands.
TASK_PROMPT_PARTS: dict[str, tuple[str, str, str]] = {
    "snapshot_counting": (_ROLE_SNAPSHOT_COUNTING, _FORMAT_BLOCK_COUNT,
                          _SEMANTICS_SNAPSHOT_COUNTING),
    "cumulative_counting": (_ROLE_CUMULATIVE_COUNTING, _FORMAT_BLOCK_COUNT,
                            _SEMANTICS_CUMULATIVE_COUNTING),
    "dedup_counting": (_ROLE_DEDUP_COUNTING, _FORMAT_BLOCK_COUNT,
                       _SEMANTICS_DEDUP_COUNTING),
    "realtime_state_monitor": (_ROLE_REALTIME_STATE_MONITOR, _FORMAT_BLOCK_STATE,
                               _SEMANTICS_REALTIME_STATE_MONITOR),
    "event_narration": (_ROLE_EVENT_NARRATION, _FORMAT_BLOCK_BASE,
                        _SEMANTICS_EVENT_NARRATION),
    "sequential_step_instruction": (_ROLE_SEQUENTIAL_STEP_INSTRUCTION,
                                    _FORMAT_BLOCK_BASE,
                                    _SEMANTICS_SEQUENTIAL_STEP_INSTRUCTION),
}


# ---------------------------------------------------------------------------
# Per-task WRITER prompts (probe-gate arm). Answer FORMAT is task knowledge
# both architectures get; the architecture is what differs.
# ---------------------------------------------------------------------------
TASK_WRITER_PROMPTS: dict[str, str] = {
    "explicit_target_grounding": _ETG_WRITER_PROMPT,
    "realtime_state_monitor": _RSM_WRITER_PROMPT,
}


# ---------------------------------------------------------------------------
# Generic / probe-gate templates (moved out of the dataclass so that ALL
# prompt text lives in this module). {placeholders} are filled at runtime.
# ---------------------------------------------------------------------------
CONTROLLER_PROMPT_GENERIC = (
    "\nYou are the CONTROLLER of a live video monitor for the target event in your "
    "task. Each turn you read the stream so far (above) and emit exactly ONE control "
    "command as a compact JSON object with these fields:\n"
    '  seen              : ALWAYS FIRST — 3-8 words: what is on screen now that is relevant\n'
    "  have_enough_info : true ONLY when you are confident the target event has occurred\n"
    "  event_time_s      : when have_enough_info is true -> the video time in seconds when THIS "
    "occurrence appeared (read it off the 'time Xs' markers in the stream)\n"
    '  answer           : REQUIRED whenever have_enough_info is true -> ONE sentence describing the REAL event; else ""\n'
    "  fps              : how densely to sample next (1-3; raise when the scene is busy)\n"
    "  next_check_s     : seconds until the next check (small when unsure or busy, larger when idle)\n"
    '  question_for_next : if unsure but something may be developing -> a short yes/no check to verify next time; else ""\n'
    "fps, next_check_s and question_for_next are asked EVERY tick — always give your "
    "current best value, even when unchanged.\n"
    "Rules: describe only what you actually see; NEVER copy the example text; NEVER "
    "re-report an event you already reported; when unsure, DEFER with a question_for_next and a "
    "short next_check_s, then confirm on the next turn before answering.\n"
    "Worked examples (Situation -> Control):\n"
    "# nothing relevant on screen yet -> stay quiet, sample slowly\n"
    "# something may be starting but you are not sure -> sample densely and DEFER a targeted check\n"
    '{"seen":"players moving toward the goal area","have_enough_info":false,"fps":3,"next_check_s":1.0,"question_for_next":"has the target event just started?"}\n'
    "# the deferred check is now clearly confirmed -> report it ONCE\n"
    '{"seen":"the target event clearly on screen","have_enough_info":true,"event_time_s":12,"answer":"<one sentence describing the real event>","fps":2,"next_check_s":1.5,"question_for_next":""}\n'
    "# the event is still on screen but you ALREADY reported it -> restate the SAME event_time_s\n"
    '{"seen":"the same event still on screen","have_enough_info":true,"event_time_s":12,"answer":"<the same sentence>","fps":1,"next_check_s":1.5,"question_for_next":""}\n'
    "Now emit ONLY your control JSON for the current stream:\n")

GOAL_QUESTION = (
    "\nQuestion: Based on the most recent frames, has this happened: {event}? "
    "Answer yes or no: ")

WRITER_PROMPT = (
    "\nThe monitored event just occurred on screen. In ONE sentence UNDER 25 "
    "words state WHAT happened AND WHY it satisfies the condition: {event}. "
    "Output only that sentence.")

SYSTEM_PROMPT = (
    "You are a helpful assistant watching a live video stream. "
    "According to the video you are watching, your task is: {instruction}")

WRITER_CUE = "\nALERT: "
