# F26-13 — Phase 5 report: Output analysis

**Project:** Confidence-Aware Self-Prompting for Robust Polyp Segmentation Under Image and Domain Shifts · Track A. Submission source: `docs/submissions/phase5/report_source.html`; notebook `notebooks/07_phase5_analysis.ipynb`.

**Summary.** Phases 3–4 showed that our method CASP v1 (**C**) beats the YOLO → SAM 2 baseline (**B**) \[1\]–\[3\] on Kvasir-SEG \[4\], on CVC-ClinicDB \[5\] (another hospital) and on nine image corruptions, but that the whole gain comes from its confidence gate: lowering the threshold alone (**T**) does as well, and SAM 2's score never rejected a box. Phase 5 asks *why*, using only the saved Phase 3 logs (1,175 candidate boxes with their YOLO confidence and SAM 2 score, ground-truth boxes from the oracle run **O**; no GPU, no Phase 3 number changed). We (i) decompose the remaining error, (ii) test which signals can tell a real polyp box from a false one, (iii) explain the results theoretically and (iv) test a rule fixed in advance from that theory, the **fallback rule F** (DL-44). Code: `src/eval/filter_analysis.py`; tables: `docs/tables/phase5/`; decisions DL-44 and DL-45.

## 1. Experimental results

**To be written by Kashish Fatima (task 1.2).** The two results tables (B, T, C, F, O on Kvasir-SEG test and CVC-ClinicDB: Dice with CI, pixel precision / recall, misses, extra-box images, box P / R / F1), the paired differences and one paragraph on the corrupted settings, from `docs/tables/phase5/fallback_metrics.csv` and `fallback_comparison.csv`.

## 2. Error analysis

**To be written by Khushbakht Sohail (task 2.4, 2.5).** Mean Dice = P(mask) × E\[Dice \| mask\] table and the oracle-gap split (numbers in notebook section 2), how F changes the failure taxonomy (Figure 2, `docs/figures/phase5_fallback_taxonomy.png`), and where F loses (ClinicDB second polyps / fragments, noise severity 2).

## 3. Theoretical insights

### 3.1 Domain shift lowers detector confidence before it removes detections

A detector's confidence is calibrated, at best, on its training distribution; under dataset shift its scores drift and usually drop \[8\]. Our logs show exactly this: the median confidence of true polyp boxes falls from 0.873 on Kvasir-SEG to 0.794 on ClinicDB, the share of true boxes at ≥ 0.5 falls from 88.6% to 77.3%, and the share below 0.25 rises from 3.0% to 12.0%. A threshold of 0.5 chosen on the source domain therefore loses recall precisely where it matters, and lowering it to 0.25 moves the operating point back along the precision–recall curve. Whether that is worth it is a cost question: a box should be kept when P(polyp \| box) × gain from a found polyp exceeds (1 − P) × cost of a false alarm \[9\]. Between 0.25 and 0.5, P(polyp) is about 0.44 on Kvasir-SEG (17 of 39 boxes) and 0.52 on ClinicDB (61 of 117); in screening, where a missed polyp is the costlier error, keeping such boxes is the right decision on average. This is the whole of CASP's gain over B.

### 3.2 Why SAM 2's score cannot filter false boxes

SAM 2's IoU head is trained to predict the IoU between the mask it outputs and the object the *prompt* designates \[3\], \[6\]. It answers "is this a clean, complete segment of whatever is in the box?", not "is the thing in the box a polyp?". A fold, a light reflection or a bubble inside a false box is a well-defined object, so SAM 2 segments it cleanly and scores it high. The data agree (Figure 1): among the kept boxes, 91% (Kvasir-SEG) and 96% (ClinicDB) of the false ones score ≥ 0.8, and inside the 0.25–0.5 band the SAM 2 score separates true from false boxes no better than chance (AUROC 0.53 \[0.34, 0.71\] and 0.47 \[0.37, 0.58\]). Over all kept boxes it does reach 0.69 and 0.66, but only because it is correlated with YOLO's confidence (r = 0.36 and 0.44): multiplying the two gives 0.861 and 0.822, identical to confidence alone (0.861, 0.820). SAM 2's score adds no information about *class*, which is why the acceptance rule of Phase 3 never fired at any threshold we could set. The same argument explains why the candidate prompts and self-prompting were neutral: for a correct box SAM 2 is already near the oracle (E\[Dice \| mask\] 0.95 with ground-truth boxes), and choosing the best of several highly correlated candidates by a noisy estimate of quality mostly selects the candidate whose score is over-estimated, so the expected gain is close to zero and can be negative \[7\].

<figure>
<img src="figures/phase5_box_signals.png" class="fig" />
</figure>

**Figure 1.** Every box T kept with confidence 0.25–0.5, labelled against the ground-truth boxes. *Left, centre:* true polyps (blue dots) and false boxes (orange crosses) overlap completely in YOLO confidence and in SAM 2's predicted IoU. *Right:* whether the image already has a box ≥ 0.5 separates them (AUROC 0.77 on Kvasir-SEG, 0.68 on ClinicDB).

### 3.3 The missing information is context, and it leads to the fallback rule

If neither per-box score carries the class, the information has to come from the rest of the image. Most colonoscopy frames show one polyp (194 of 200 Kvasir-SEG and 582 of 612 ClinicDB test images), so by Bayes' rule an uncertain box is far less likely to be a polyp when a confident one has already been found: in our logs 7% (1 of 14) versus 64% (16 of 25) on Kvasir-SEG and 27% (11 of 41) versus 66% (50 of 76) on ClinicDB (Figure 1, right). This is the reasoning behind F, written down before computing its results (DL-44) and using only the existing thresholds 0.25 and 0.5, so nothing was tuned on the test sets. F acts as an image-level suppression step with a single-polyp prior. Its limits follow from the same prior: it removes real second polyps and fragments of large polyps (ClinicDB box recall 0.780 → 0.763), and under strong noise, where the confident box itself is more often wrong, the uncertain box it discards is sometimes the polyp (noise severity 2). The 75 ClinicDB images with no box above 0.25 are out of reach of any rule that only re-ranks YOLO's boxes; they need a detector that transfers better (training on both hospitals, or test-time adaptation).

## 4. Functionality and reproducibility

**To be written by Khushbakht Sohail (task 2.6).** What the hosted demo shows (with its link and a screenshot), how to rerun the analysis (`python src/eval/filter_analysis.py --runs runs`; the notebook first reproduces the Phase 4 tables byte for byte) and the CI tests.

## 5. Conclusions

1.  CASP's improvement is a recall effect of the confidence gate, and it is explained by domain shift lowering detector confidence (§3.1); the refinement steps cannot add to it because SAM 2's score measures mask quality, not class (§3.2).
2.  A context rule fixed in advance (F) keeps all of the gate's recall, matches T and C in Dice and removes about half of their false boxes on the unseen hospital, at the price of some second polyps and of robustness under strong noise.
3.  The remaining error on the training hospital is box quality; on the unseen hospital it is mostly polyps the detector never proposes. Further gains must come from the detector, not from prompting SAM 2 differently.

**Work division** (`project-division/Project-Division-Phase-5.md`): Kashish Fatima (lead): §1, decision log, review. Khushbakht Sohail: hosted demo, §2, §4. Zahra Saeed: filter analysis, notebook, §3, summary and conclusions, package.

### References

1.  M. Mansoori, S. Shahabodini, J. Abouei, K. N. Plataniotis, and A. Mohammadi, “Self-prompting polyp segmentation in colonoscopy using hybrid YOLO-SAM 2 model,” 2024, *arXiv:2409.09484*.
2.  G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics YOLOv8,” 2023. \[Online\]. Available: https://github.com/ultralytics/ultralytics
3.  N. Ravi *et al.*, “SAM 2: Segment anything in images and videos,” 2024, *arXiv:2408.00714*.
4.  D. Jha *et al.*, “Kvasir-SEG: A segmented polyp dataset,” in *MultiMedia Modeling (MMM 2020)*, LNCS, vol. 11962. Cham, Switzerland: Springer, 2020, pp. 451–462.
5.  J. Bernal *et al.*, “WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians,” *Comput. Med. Imag. Graph.*, vol. 43, pp. 99–111, 2015.
6.  A. Kirillov *et al.*, “Segment anything,” in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2023, pp. 4015–4026.
7.  J. E. Smith and R. L. Winkler, “The optimizer's curse: Skepticism and postdecision surprise in decision analysis,” *Manage. Sci.*, vol. 52, no. 3, pp. 311–322, 2006.
8.  Y. Ovadia *et al.*, “Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift,” in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019.
9.  C. Elkan, “The foundations of cost-sensitive learning,” in *Proc. Int. Joint Conf. Artif. Intell. (IJCAI)*, 2001, pp. 973–978.
