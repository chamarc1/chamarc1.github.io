# Charlemagne Marc Portfolio Site

Professional portfolio site hosted with GitHub Pages.

## Overview

This repository contains a static HTML/CSS/JS portfolio featuring:

- ESI (Earth and Space Institute)
- Flatfield project
- SWIR L1A project
- Coaching
- Education
- About
- Contact

Primary entry point: `index.html`

## Project Structure

```text
chamarc1.github.io/
├── index.html
├── assets/
│   ├── css/
│   ├── js/
│   ├── sass/
│   └── webfonts/
├── files/
│   ├── Flatfiled/
│   └── swir-l1agen/
└── images/
```

## Run Locally

From the repository root:

```bash
python3 -m http.server 8000
```

Open in a browser:

- `http://localhost:8000`

## Content Updates

- Update page content in `index.html`
- Update styles in `assets/css/main.css` or SASS sources in `assets/sass/`
- Add or update project artifacts under `files/`
- Add or update images under `images/`

### Flatfield and SWIR L1A Sections

Both project sections in `index.html` use the same structure:

1. Tech Stack
2. Environment
3. Project Artifacts
4. Source Files
5. Directory Structure

## Deployment

The site is published through GitHub Pages from the default branch.

Typical publish flow:

```bash
git add .
git commit -m "Update portfolio content"
git push
```

After push, GitHub Pages publishes the updated site automatically.

## Website Tracking (Location + Visit Time)

Google Analytics 4 has been added to `index.html` with a placeholder measurement ID.

Setup steps:

1. Create a GA4 property and Web Data Stream for `https://chamarc1.github.io`
2. Copy your Measurement ID (format: `G-XXXXXXXXXX`)
3. In `index.html`, replace both `G-XXXXXXXXXX` values with your real ID
4. Push changes to GitHub Pages

What you can see in GA4:

- Visit date/time trends (Reports and Realtime)
- Visitor location by country/city (aggregated by IP-derived geo)
- Device, browser, and traffic source details

Note: if you use exact browser geolocation or any personally identifiable tracking, add an explicit consent banner and privacy-policy disclosure to stay compliant with privacy laws.

## Quality Checklist

Before each push:

- Verify all links in `index.html` resolve correctly (artifacts, images, and external references)
- Confirm any renamed files in `files/` or `images/` are updated in `index.html`
- Ensure Flatfield and SWIR L1A retain the same block order and label style
- Confirm newly added images render at the expected size and aspect ratio
- Verify downloadable artifacts open from the deployed site path
- Review heading capitalization and punctuation for consistency
- Run a local preview (`python3 -m http.server 8000`) and perform a full-page visual pass

## Release Notes

Use this template for each update:

```markdown
### YYYY-MM-DD

#### Added
- 

#### Changed
- 

#### Fixed
- 

#### Validation
- Local preview checked at `http://localhost:8000`
- Key links/artifacts verified
```

### 2026-03-06

#### Added
- Flatfield section source file links and directory structure block in `index.html`
- SWIR L1A section source file links and directory structure block in `index.html`
- Root `README.md` for site overview, structure, local preview, and deployment steps
- Content maintenance checklist and release notes template in `README.md`

#### Changed
- Normalized Flatfield and SWIR L1A artifact/source ordering for consistent presentation
- Standardized heading capitalization and punctuation across updated project areas
- Expanded Tech Stack lines to reflect actual libraries/tech used in both projects
- Added reproducibility-focused Environment lines for Flatfield and SWIR L1A

#### Fixed
- Standardized artifact label wording across both project sections

#### Validation
- HTML diagnostics run with no errors in `index.html`
- Internal link/file path updates verified during edit pass

## Notes

- Keep links in `index.html` relative so artifacts and images resolve correctly on GitHub Pages.
- If files are renamed in `files/` or `images/`, update all corresponding links in `index.html`.
