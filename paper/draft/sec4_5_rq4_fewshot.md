# 4.5 RQ4: Few-Shot Target Calibration

Week9 is a post-hoc calibration layer. Its purpose is not to show that few-shot labels universally improve transfer, but to test whether a hard but recoverable target can be repaired with a small number of balanced target labels.

The frozen Week10 few-shot table uses the formal W9 outputs rather than the older preliminary files. For T2_PH, few-shot calibration is consistently beneficial. Requested 10-shot calibration is capped by the available target adaptation pool, but it still gives a large gain under full features. The expanded feature-condition audit further shows that T2_PH improvement persists under no-ccTLD, no-website-lexical, and graph-only settings.

T3 France has a different interpretation. Its source-only performance is already saturated and Week8 shows lexical/ccTLD assistance, so small additional gains under few-shot should not be treated as a few-shot success or as structural generalization evidence.

The paper-safe claim is: few-shot local calibration can repair a hard but recoverable target such as T2_PH, while offering little meaningful additional value for saturated shortcut-sensitive targets such as T3 France.

Forbidden interpretation: do not claim few-shot universally improves transfer.
