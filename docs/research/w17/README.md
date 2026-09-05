W17 — owner annotation round-trip for the granular timeline study.

The owner marks, by ear and eye and BEFORE seeing any machine output, where a
demonstration is actually in tempo and when he would commit as accompanist.
This script emits an Audacity label template for that pass and reads the
finished file back.

    python scripts/w17-owner-annotation.py emit  --clip barre6-frappe-demo
    python scripts/w17-owner-annotation.py read  --clip barre6-frappe-demo

Audacity label format is tab-separated `start<TAB>end<TAB>text`; a point label
has start == end. Vocabulary (only the first two are required):

  fullout           REGION - teacher dancing it full-out, genuinely in tempo
  commit            POINT  - the moment you would commit to a tempo
  tempo=<bpm>       POINT or REGION - the tempo you would commit to
  marking           REGION - sketching the combination, not in tempo
  talking           REGION - explaining; no movement tempo to read
  cue=voice|feet|arm|breath|other   REGION - what you are reading tempo from

Unknown labels are preserved rather than rejected: the vocabulary is a
starting point and the owner may need words it does not have.
