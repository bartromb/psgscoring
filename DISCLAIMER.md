# Medical and Clinical Disclaimer

**psgscoring** — AASM 2.6-compliant respiratory event scoring for polysomnography  
Copyright (c) 2024–2026 Bart Rombaut / Slaapkliniek AZORG  
https://github.com/bartromb/psgscoring

---

## 1. Research Software — Not a Medical Device

`psgscoring` is **research software**. It is intended exclusively for use by
qualified professionals (physicians, researchers, registered polysomnographic
technologists, or biomedical engineers) in a **research or clinical research
context**.

This software has **not** been evaluated, cleared, or approved by any
regulatory authority as a medical device, including but not limited to:

- The European Union Medical Device Regulation (EU MDR 2017/745)
- The U.S. Food and Drug Administration (FDA 21 CFR Part 820 / 510(k))
- Any equivalent national or regional medical device framework

It does **not** carry a CE mark, FDA clearance, or any equivalent certification.

---

## 2. Not a Substitute for Clinical Judgement

The computed indices produced by this software — including but not limited to
the Apnoea-Hypopnoea Index (AHI), Obstructive AHI (OAHI), Oxygen Desaturation
Index (ODI), Periodic Limb Movement Index (PLMI), and arousal index — are
**research-grade estimates**. They must be:

- Reviewed by a qualified, licensed clinician before any diagnostic or
  therapeutic decision is made.
- Validated against manual polysomnographic scoring by a registered
  polysomnographic technologist (RPSGT) for any clinical application.
- Interpreted in the context of the full clinical picture, patient history,
  and concurrent PSG signals.

**This software does not provide medical diagnoses, treatment recommendations,
or any form of clinical advice.**

---

## 3. Known Limitations

Users should be aware of the following limitations that may affect scoring
accuracy:

| Condition | Effect |
|-----------|--------|
| Mouth-breathing | Reduced nasal flow → hypopnoea under-detection or false positives |
| Poor RIP-belt contact | Unreliable effort signals → apnoea type misclassification |
| Very high AHI (> 60 /h) | SpO₂ cross-contamination between consecutive events |
| Cheyne-Stokes respiration | Decrescendo phases may be scored as hypopnoeas |
| Signal dropout / sensor displacement | Post-gap recovery ramp may be scored as event |
| Paediatric recordings | Not validated for patients under 18 years of age |
| Non-AASM sensor configurations | Results may deviate from manual AASM 2.6 scoring |

A pilot validation study comparing `psgscoring` output against consensus RPSGT
scoring (target n = 50) is in preparation. Until published, results should be
treated as provisional.

---

## 4. No Warranty

This software is provided **"as is"**, without warranty of any kind, express
or implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose, and non-infringement.

---

## 5. Limitation of Liability

To the fullest extent permitted by applicable law, in no event shall the
authors, contributors, or Slaapkliniek AZORG be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages, including
but not limited to:

- Patient harm or adverse clinical outcomes
- Diagnostic errors or missed diagnoses
- Loss of data or corrupted results
- Business interruption or financial loss

arising from the use of, or inability to use, this software — even if advised
of the possibility of such damages.

---

## 6. User Responsibility

By installing or using `psgscoring`, the user confirms that they:

1. Are a qualified professional with appropriate training in sleep medicine,
   polysomnography, or biomedical engineering.
2. Will not use the output of this software as the sole basis for any clinical
   decision without independent clinical review.
3. Accept full responsibility for validating the software's output in their
   specific recording environment, patient population, and clinical workflow.
4. Will comply with all applicable local laws and regulations regarding the
   use of software in medical and research settings.

---

## 7. Contact

For questions, bug reports, or validation data:

- GitHub Issues: https://github.com/bartromb/psgscoring/issues
- Clinical context: Slaapkliniek AZORG, Aalst, Belgium
- Live instance: https://www.slaapkliniek.be

---

*Last updated: 2026 — psgscoring v0.2.0*
