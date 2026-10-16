# Demo site

Static page (`index.html`) deployed to GitHub Pages by `.github/workflows/pages.yml`.
Preview locally with `python -m http.server -d demo`.

The framework figure is drawn by `figure.py` into `framework.svg` (inlined and
animated on the page). `framework.png` is a 2x static render of it for the README:
inline the SVG in a page, open it at 1600 × 650 and take a screenshot at device
scale 2.

## Listening examples

`audio/` holds 30 s excerpts (from 0:45 unless noted below) of vocal-removed Jamendo tracks from the ATTM
challenge test set, under the artists' Creative Commons licences (no ND licences, since
the audio is vocal-removed). `midi/` holds three transcriptions of each excerpt, all with
MuScriptor large: `amtflow` (`--config paper`), `no-tags` (`tagger: {name: fixed,
families: []}`) and `all-tags` (all seven families fixed). The tracks were picked from
random test tracks with Jamendo instrument tags: in both baselines many notes fall in
families the tags do not list, or tagged families are missing, while AMTFlow's tracks
match the tags. Jamendo tags are often incomplete (many electronic tracks are tagged only
`synthesizer`), so candidates were checked by hand.

The Brahms waltz (1169734, 0:15–0:45) and Sourire Fragile (88558, 0:35–1:05) show
MuScriptor with all tags writing instruments a drumless piano or chamber piece does not
have (guitar, drums, bass, synth pad). They come from the same full-track transcriptions
as Dancing Santa below: tracks whose Jamendo tags list no drums or synthesizer were ranked
by the share of all-tags notes outside the tagged families, and kept only when AMTFlow's
excerpt stays within them.

Dancing Santa (632094) instead shows a full-track failure: all three versions were
transcribed from the whole track and cut to 0:55–1:25. It was found by transcribing
167 full test tracks with all tags and looking for blow-ups (many notes stacked on one
onset, held notes, note bursts), then checking those tracks without tags and with
AMTFlow. The blow-ups mostly appear late in a track, often in its last 5 s chunk.
