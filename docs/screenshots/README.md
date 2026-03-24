# docs/screenshots

This folder contains annotated screenshots and captions used in the main README and presentation materials.

Naming convention:
- `dashboard_overview.png` — Dashboard grid and zone cards
- `zone_fullscreen_<zoneId>.png` — Fullscreen view for zone `<zoneId>` with detection overlays
- `alert_timeline_<lg_id>_<timestamp>.png` — Snapshot captured at alert time with caption and context
- `lifeguards_tab.png` — Lifeguards tab showing registered chat IDs and Stop/Resume buttons

For each image add a small text file with the same base name and `.caption.md` extension containing:
- Title (one line)
- Short caption (1–2 sentences)
- Alt text (one line)

Example file pairs:
- `dashboard_overview.png`
- `dashboard_overview.caption.md`

Accessibility & formats:
- Use PNG for screenshots; SVG is allowed for diagrams.
- Keep images under 2 MB for GitHub readability; for larger demo videos host externally and link here.

Demo video:
- If you add `docs/demo.mp4` or host on YouTube, add a `demo.link.md` with the external URL and timestamps for key clips.

How to add:
1. Put the image in this folder.
2. Create the corresponding `.caption.md` file.
3. Commit the image and caption.

If you want, I can create annotated templates or add a `placeholder_dashboard.svg` to serve as an example image. Let me know if you want that.
