# Tell pytest not to collect/import the root-level __init__.py as a test module.
# The root __init__.py is the v0.2.0 development package; it uses relative imports
# that only work when imported as part of a proper package, not standalone.
collect_ignore = ["__init__.py"]
