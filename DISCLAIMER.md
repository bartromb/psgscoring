# DISCLAIMER

## Not a Medical Device

**psgscoring and YASAFlaskified are research software — not medical devices.**

This software has **not** been evaluated, cleared, or approved by any
regulatory authority, including the European Union Medical Device Regulation
(EU MDR 2017/745) and the U.S. Food and Drug Administration (FDA). It does
**not** carry a CE mark or FDA clearance.

## Not a Substitute for Clinical Judgment

All computed indices — including AHI, OAHI, ODI, PLMI, arousal index,
RDI, and severity classifications — are **research-grade estimates**.
They must be:

- **Reviewed** by a qualified, licensed clinician before any diagnostic
  or therapeutic decision
- **Validated** against manual scoring by a registered polysomnographic
  technologist (RPSGT) or experienced sleep physician
- **Interpreted** in the context of the full clinical picture and
  patient history

Automated scoring is not a replacement for expert clinical judgment.

## Intended Use

| ✓ Designed for | ✗ NOT designed for |
|---|---|
| Academic sleep research | Standalone clinical diagnosis without expert review |
| Algorithm development and benchmarking | Automated treatment decisions |
| Clinical research under ethics committee approval | Unsupervised patient screening programmes |
| Educational purposes in sleep medicine | Any setting where output directly determines patient care |
| Second opinion and clinical decision support | |

## Validation Status

External validation on the PSG-IPA dataset (PhysioNet, 5 recordings,
59 independent scorer sessions) demonstrated mean |ΔAHI| = 2.0/h and
severity concordance 4/5 (80%). External validation on the PSG-Audio
dataset (Sismanoglio Hospital, Athens, n=194) is in progress.

A formal single-centre validation study (AZORG-YASA-2026-001, n≥50,
stratified by severity, Bland-Altman, weighted κ) is in preparation.

Until peer-reviewed validation results are published, all outputs
should be treated as **preliminary** and verified by a qualified
clinician.

## No Warranty

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

*psgscoring v0.2.91 · YASAFlaskified v0.8.34 · April 2026*
*Slaapkliniek AZORG, Aalst, Belgium*
