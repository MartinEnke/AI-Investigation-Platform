# Repository Guidelines

## Project Structure & Module Organization

This repository is currently an empty project scaffold. As implementation begins, keep production code under `src/`, automated tests under `tests/`, documentation under `docs/`, and static or sample assets under `assets/`. Group code by feature or domain rather than by file type; for example, use `src/investigations/` for investigation workflows and mirror that path in `tests/investigations/`. Keep generated output in ignored directories such as `dist/`, `build/`, or `coverage/`.

## Build, Test, and Development Commands

No build system or package manager has been configured yet. When adding one, expose a small, predictable command set and document it in the root `README.md`. Prefer conventional commands such as:

- `npm run dev` — start the local development environment.
- `npm run build` — create a production build.
- `npm test` — run the complete automated test suite.
- `npm run lint` — check formatting and static-analysis rules.

Do not commit dependencies or generated build artifacts.

## Coding Style & Naming Conventions

Follow the formatter and linter selected by the first implementation change, and commit their configuration at the repository root. Use spaces rather than tabs unless the language ecosystem dictates otherwise. Choose descriptive names: `camelCase` for variables and functions, `PascalCase` for classes or UI components, and `kebab-case` for directories and general-purpose filenames. Keep modules focused and avoid hidden side effects.

## Testing Guidelines

Every feature or bug fix should include tests once a test framework is established. Mirror source paths in `tests/`, and use names such as `investigation-service.test.ts` or the equivalent convention for the chosen language. Cover normal behavior, validation failures, and important edge cases. Run the full suite before opening a pull request.

## Commit & Pull Request Guidelines

There is no existing commit history to establish a local convention. Use concise, imperative subjects, optionally following Conventional Commits (for example, `feat: add evidence ingestion`). Keep commits narrowly scoped. Pull requests should explain the problem and solution, list verification steps, link relevant issues, and include screenshots for visible UI changes. Call out configuration changes, migrations, or security implications explicitly.

## Security & Configuration

Never commit secrets, credentials, private evidence, or real personal data. Store local settings in ignored `.env` files and provide a sanitized `.env.example` when configuration is introduced.
