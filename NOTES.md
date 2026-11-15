# Project Part 1 notes

Working alone, so: two vision systems, no 3D scenes required.
Deadline: 20 November (strict), upload in the Studon course.

## Tasks

1. Choose an image type and select or generate 100 images.
2. Tool (GUI or CLI) with controls for blur strength and percentage of white
   pixels, able to batch-process 100+ images. Generate 100 Mooneys, pick the
   top 10 by hand.
3. Run all images through two vision systems, as originals and as Mooneys.

## Forum rulings (Bernhard Egger)

- Use pretrained models. No fine-tuning: "the point is generalization".
- "Different vision systems" may be different tasks or different models for
  the same task. Different datasets per task are allowed.
- The 100 Mooneys are generated automatically with the same blur and white %.
  That is why the best 10 are then picked by hand.
- Blur can stay constant if resolution and content are similar.
- Any automatic thresholding is fine. Binary search is not required; a
  histogram or percentile is an accepted shortcut.

## Zip structure

```
Original/            100 original images
Mooney/              100 Mooney images
Subset/Original/     the selected 10 originals
Subset/Mooney/       the selected 10 Mooneys
Result1/Original/    first vision system, results for all originals
Result1/Mooney/      first vision system, results for all Mooneys
Result2/Original/    second vision system
Result2/Mooney/
slides.pptx
```

Keep the zip well under 1 GB. If the upload fails, send the md5 hash of the zip
before the deadline and a download link within 24 hours.

## Slides, in this order

1. Title slide with all team member names
2. One-page description of the approach
3. The 10 original and 10 Mooney images
4. Declaration of the vision systems used
5. Results for each vision system
6. Screenshot of the tool
7. Screenshot of the code that generates Mooneys
8. Declaration of AI usage

## Grading

Five items: folder structure and files, slide structure, Mooney generation as
described, results on Mooneys, results on originals. Pass allows at most one
to be missing; the selected best solution fulfils all five.

## Decisions so far

- Question: do vision systems still see what is in a Mooney image?
- System 1: a vision-language model (Qwen3-VL), scored with questions a script
  can check.
- System 2: SAM 3, prompted with the object's name. Fallback if it cannot be
  installed in time: a monocular depth model.
- One Mooney setting, chosen by eye and fixed for all 100 images. It is not
  tuned to make the models score well.
- Add-ons (polarity flip, upside-down, parameter sweep) come after the
  required deliverables are complete, and go in the slides only.
