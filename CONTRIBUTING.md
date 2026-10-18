# Contributing to PegaInfer

First off, thank you for considering contributing to our project! It's people like you that make this community great.

To get you started quickly, here is everything you need to know to get your first Pull Request (PR) reviewed.

## 💬 Where to Ask Questions

If you have questions, need help, or want to discuss a feature, please reach out via:

* **GitHub Issues & Discussions:** Please use this repository's official Issues and Discussions tabs to ask questions or report problems.

## 🛠️ Setup & Development

We keep our detailed technical documentation close to the code. Please refer to these guides:

* **First-time Setup:** See [Developer Onboarding](docs/playbooks/developer-onboarding.md) to set up your toolchain.

* **Building & Running Tests:** Check [CLAUDE.md](./CLAUDE.md) for quick commands on building and running the test suite.

* **Coding Style & Principles:** Read our [Coding Style Guide](docs/conventions/coding-style.md).

## 🌿 Branch Naming

When creating a branch, use the following prefixes depending on your work:

* `feat/` for new features (e.g., `feat/add-login-button`)

* `fix/` for bug fixes (e.g., `fix/memory-leak`)

* `chore/` for maintenance, CI/CD, or dependency updates (e.g., `chore/update-deps`)

## 📝 Commit Messages

We use **Commitizen** conventions for clear history. Your commit messages should look like this:

`type(scope): description` (e.g., `feat(auth): add OAuth2 support`)

## 🚀 How to Submit a PR

1. Fork the repository and create your branch from `main`.

2. Ensure your code passes all tests (e.g., cargo test --release).

3. Open a Pull Request against the `main` branch.

4. Fill out the PR template—our maintainers will review it shortly!
