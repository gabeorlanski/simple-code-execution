# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets for common tasks
.PHONY: clean-all install-deps serve

clean-all:
	rm -rf $(BUILDDIR)/*

install-deps:
	pip install -r requirements.txt

serve: html
	@echo "Starting development server..."
	@echo "Documentation will be available at http://localhost:8000"
	@cd $(BUILDDIR)/html && python -m http.server 8000

live:
	@echo "Starting live reload server..."
	@sphinx-autobuild $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS) 