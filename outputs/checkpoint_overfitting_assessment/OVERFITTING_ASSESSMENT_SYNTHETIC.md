OVERFITTING ASSESSMENT -- SYNTHETIC DATA RESULTS ONLY
========================================================
Checkpoints assessed: OHMIC, FIC, SIN, THIN. No model was trained, tuned, or
reselected to produce this assessment; it only loads the pipelines already
locked in outputs/checkpoint_training/ and scores them against the exact
original training partition (1200 rows), the already-saved
grouped-CV fold results, and the already-saved test-set metrics.

Per-checkpoint RMSE gaps:
- OHMIC: Train=0.1927, CV=0.6102 (+/-0.0854), Test=0.4716. Train-vs-CV gap=0.4175; Train-vs-Test gap=0.2789.
- FIC: Train=0.0382, CV=0.3719 (+/-0.0346), Test=0.3214. Train-vs-CV gap=0.3337; Train-vs-Test gap=0.2832.
- SIN: Train=0.0140, CV=0.3747 (+/-0.0371), Test=0.3111. Train-vs-CV gap=0.3607; Train-vs-Test gap=0.2971.
- THIN: Train=0.0438, CV=0.3081 (+/-0.0303), Test=0.2480. Train-vs-CV gap=0.2643; Train-vs-Test gap=0.2042.

What these numbers do and do not mean
--------------------------------------
1. Train scores are in-sample: the locked pipeline was fit on these exact
   rows, so its Train score reflects memorization capacity as well as true
   signal. A Train score better than CV/Test is EXPECTED for any model with
   nonzero capacity and is not, by itself, evidence of problematic
   overfitting -- it is evidence the model fit the training rows, which is
   what fitting means.
2. CV models were fit on fewer rows than the locked model. Each of the 5
   outer-CV folds trained on roughly 4/5 of the 80-lot train partition,
   not all of it. A gap between CV and Test is therefore not fully
   comparable to a Train-vs-Test gap: part of any CV-vs-Test difference can
   come from training-set size alone, not only from how the final,
   full-data-trained model generalizes.
3. CV fold-to-fold standard deviation (shown as the CV error bar) describes
   the spread actually observed across 5 folds. It is NOT a confidence
   interval, is not based on a distributional assumption, and should not be
   read as "the true RMSE lies within +/- 1 std with 95% probability" --
   with only 5 folds the standard deviation itself is a noisy estimate.
4. These plots and this table do not, on their own, prove the absence of
   overfitting. They describe the gaps that exist in this one synthetic run
   and offer context for interpreting them; they cannot rule out that the
   locked models would perform differently on data drawn from a materially
   different distribution (in particular: real production data).
5. No model selection, hyperparameter, or pipeline choice was changed in
   response to any number in this assessment, including the Test numbers
   above, which were already fixed before this assessment began and remain
   fixed after it.

All results above are SYNTHETIC DATA RESULTS (Process == 'GaAs_pHEMT_MMIC').
No production-accuracy or real-data-compatibility claim is made or implied.
