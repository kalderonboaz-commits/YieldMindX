# Shared packages

contracts/ holds the chosen API schema source and generation setup. ui/ is optional when multiple consumers actually need shared UI. Keep database models and server-only dependencies out of browser packages. Empty packages do not require a build system yet.
