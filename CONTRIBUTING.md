# Contributing to AI News Podcast

First off, thank you for considering contributing to `ai-news-podcast`! It's people like you that make open source such a great community.

## Where do I go from here?

If you've noticed a bug or have a question, please search the [issue tracker](https://github.com/nicekai-jpg/ai-news-podcast/issues) to see if someone else has already created a ticket. If not, go ahead and make one!

## How to Contribute

We welcome all contributions, from bug fixes to new features. 

### 1. Fork & Create a Branch
1. Fork the repository.
2. Create a new branch: `git checkout -b my-new-feature`
3. Make your changes.

### 2. Follow the Development Guide
Please refer to our [`DEVELOPMENT.md`](DEVELOPMENT.md) for instructions on setting up the local environment (using `uv`), project structure, and running tests.

### 3. Commit your Changes
Commit your changes with a descriptive message.
```bash
git commit -m "feat: added new RSS source for AI research"
```

### 4. Push and create a Pull Request
1. Push to your branch: `git push origin my-new-feature`
2. Submit a pull request. Please describe what your changes do and reference any related issues.

## Adding New RSS Sources
If you are adding new feeds to `config/sources.yaml` or `config/sources_edu.yaml`:
- Ensure the feed is stable and actively maintained.
- Try to select feeds that provide full-text or a robust summary.
- Run `uv run pytest` to ensure you haven't broken the Fetcher's parser.

## Reporting Bugs / Asking for Features
Please use the provided GitHub Issue templates to ensure you provide enough context for us to help you.

Thanks again for your interest in improving `ai-news-podcast`!
