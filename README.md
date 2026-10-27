# Charlemagne Marc Portfolio Site

Professional portfolio site hosted with GitHub Pages.

## Overview

This repository contains a static HTML/CSS/JS portfolio organized around:

- Selected work: SWIR Flatfield calibration, SWIR Level 1A processing, and VNIR quicklook generation
- Experience: Earth and Space Institute, coaching, and education
- About, résumé, and contact links

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

### Project Case Studies

Project summaries in `index.html` focus on the problem, contribution, approach or output, selected visuals, and links to source or supporting artifacts. Keep setup instructions, full file listings, and implementation details in the linked project READMEs rather than duplicating them on the portfolio page.

The résumé is linked from the homepage and About section. Update the PDF in `files/` and its links together when publishing a new version.

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

Google Analytics 4 is configured in `index.html`.

To use a different property:

1. Create a GA4 property and Web Data Stream for `https://chamarc1.github.io`
2. Copy your Measurement ID (format: `G-XXXXXXXXXX`)
3. In `index.html`, replace both measurement ID values with the new ID
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
- Confirm homepage, case study, résumé, and contact links work
- Confirm project visuals and downloadable artifacts load at the deployed site path
- Check keyboard focus, mobile navigation, and horizontal overflow
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

### 2026-10-27

#### Changed
- Reorganized the homepage around résumé access, selected work, experience, and contact
- Reframed SWIR project content as concise case studies and moved setup/file-list details to project READMEs
- Shortened coaching and education content while retaining the existing education dates
- Added social preview metadata, descriptive image text, external-link protections, and visible keyboard focus
- Updated responsive styles for project cards and mobile case-study headings

#### Validation
- Local browser preview checked at desktop and mobile widths
- Project navigation and mobile horizontal overflow verified
- Local links, images, and external-link attributes checked
- `git diff --check` passes

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
