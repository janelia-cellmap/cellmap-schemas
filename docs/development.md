# Adding to this project

1. Clone the repository to a local path:
```
git clone https://github.com/janelia-cellmap/cellmap-schemas.git
```

2. Navigate to that directory and install dependencies. This project uses the [`poetry`](https://python-poetry.org/) package manager.
```
poetry install
```

2. Make changes.

3. Run tests with pytest:
```
pytest tests
```

4. Preview documentation changes with `mkdocs`:
```
mkdocs serve --watch src
```

5. Submit a pull request with your changes to the main code branch on [github](https://github.com/janelia-cellmap/cellmap-schemas).