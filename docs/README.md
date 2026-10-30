# docs/

This directory is a Jekyll site (the `just-the-docs` theme) that documents
brain-lib as a whole. GitHub Pages serves it from the `main` branch, `/docs`
folder; every `.md` file here with front matter is a page.

Preview locally:

```bash
cd docs
bundle install
bundle exec jekyll serve
# open http://127.0.0.1:4000/brain-lib/
```

The three original design notes (`NEUROEMITTERS.md`, `PLASTICITY_GUIDE.md`,
`GPT_GUIDE.md`) stay at their paths so existing links from the readmes keep
working; their front matter places them in the site navigation.
