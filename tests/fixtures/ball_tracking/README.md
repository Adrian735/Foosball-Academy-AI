# Ball-tracking fixtures

This directory is reserved for curated ball-tracking clips. Clips must be at
least five seconds long and 720p or higher, contain the supported diagonal
Bonzini setup, and have no audio requirement.

The current source candidates are listed in
`tests/fixtures/expected/ball_tracking_manifest.json`. They remain in
`tests/table-detection_tests/` until a human selects and copies curated clips
here. Do not generate expected coordinates from detector output.

Create an annotation with:

```bash
python tools/annotate_ball.py \
  tests/fixtures/ball_tracking/<clip>.mp4 \
  tests/fixtures/expected/ball_tracking_<name>.json
```

By default, the tool limits the annotation window to the first 400 source
frames and shows every tenth frame, so annotate only one in ten frames. The
JSON records the full source-video metadata, the frame limit, source frame
indices, and sampling stride; unlabelled frames remain unknown and are not
treated as detector ground truth. Change the limit with `--max-frames`.
Use `--sample-every 1` when dense annotation is needed.

To begin at the middle of a video, add `--start-position middle`. The
annotation still preserves the original source frame index.

Click the ball centre for visible frames. Use `h` for an intentionally hidden
or unannotatable sampled frame, `u` to undo the current frame, left/right
arrow keys or `a`/`d` to move by the configured stride, and `s` to save.
