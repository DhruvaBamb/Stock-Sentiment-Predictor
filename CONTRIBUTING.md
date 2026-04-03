# Contributing to Stock Sentiment Predictor

Thank you for your interest in contributing! Here's how to get started.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/DhruvaBamb/Stock-Sentiment-Predictor.git
cd Stock-Sentiment-Predictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install pylint flake8 black mypy bandit pip-audit pytest pytest-cov
```

## Code Quality Standards

This project uses:
- **Black** for code formatting
- **Flake8** for style checking
- **Pylint** for code analysis
- **Mypy** for type checking
- **Bandit** for security scanning
- **Pytest** for unit testing

## Before Submitting a Pull Request

1. **Format your code** with Black:
```bash
black app.py
```

2. **Run linting checks**:
```bash
flake8 app.py
pylint app.py
```

3. **Run type checking**:
```bash
mypy app.py --ignore-missing-imports
```

4. **Run security scan**:
```bash
bandit -r app.py
```

5. **Check for dependency vulnerabilities**:
```bash
pip-audit
```

6. **Run tests**:
```bash
pytest
```

## Running All Checks Locally

Use this script to run all checks before pushing:
```bash
black app.py && \
flake8 app.py && \
pylint app.py && \
mypy app.py --ignore-missing-imports && \
bandit -r app.py && \
pip-audit && \
pytest
```

## Pull Request Guidelines

- Create a feature branch from `main`
- Make your changes and ensure all tests pass
- Keep commits clear and descriptive
- Add/update tests for new functionality
- Ensure code passes all quality checks
- Write a clear PR description

## Code Style

- Maximum line length: 120 characters
- Follow PEP 8 style guide
- Use type hints where possible
- Add docstrings for classes and public methods

## Questions?

Feel free to open an issue for questions or discussions.

Happy coding!