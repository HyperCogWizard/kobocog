# OpenCog-GGML Integration Makefile
# This file integrates OpenCog cognitive components with the main Kobocog build system

# OpenCog-specific flags
OPENCOG_CFLAGS = -DOPENCOG_GGML_INTEGRATION
OPENCOG_CXXFLAGS = -DOPENCOG_GGML_INTEGRATION -std=c++17
OPENCOG_INCLUDES = -I./opencog -I./opencog/atomspace -I./opencog/operations -I./opencog/reasoning -I./opencog/attention -I./opencog/api

# OpenCog source files
OPENCOG_SOURCES = \
	opencog/atomspace/tensor_atomspace.cpp \
	opencog/operations/ggml_opencog_ops.cpp \
	opencog/reasoning/pln_tensor_engine.cpp \
	opencog/attention/ecan_attention.cpp \
	opencog/api/cognitive_api.cpp

# OpenCog object files
OPENCOG_OBJECTS = $(OPENCOG_SOURCES:.cpp=.o)

# OpenCog headers
OPENCOG_HEADERS = \
	opencog/atomspace/tensor_atomspace.h \
	opencog/operations/ggml_opencog_ops.h \
	opencog/reasoning/pln_tensor_engine.h \
	opencog/attention/ecan_attention.h \
	opencog/api/cognitive_api.h

# Default target includes OpenCog components
.PHONY: opencog-components
opencog-components: $(OPENCOG_OBJECTS)

# Individual component targets
.PHONY: atomspace reasoning attention api operations
atomspace: opencog/atomspace/tensor_atomspace.o
reasoning: opencog/reasoning/pln_tensor_engine.o  
attention: opencog/attention/ecan_attention.o
api: opencog/api/cognitive_api.o
operations: opencog/operations/ggml_opencog_ops.o

# Compile OpenCog C++ sources
opencog/%.o: opencog/%.cpp $(OPENCOG_HEADERS)
	g++ $(CXXFLAGS) $(OPENCOG_CXXFLAGS) $(OPENCOG_INCLUDES) -c $< -o $@

# Test targets
.PHONY: test-opencog
test-opencog:
	@echo "Running OpenCog integration tests..."
	cd opencog/tests && python3 test_tensor_atomspace.py

# Clean OpenCog objects
.PHONY: clean-opencog
clean-opencog:
	rm -f $(OPENCOG_OBJECTS)

# Install OpenCog components (for development)
.PHONY: install-opencog
install-opencog: opencog-components
	@echo "OpenCog-GGML components built successfully"
	@echo "Integration ready for Kobocog core"

# Integration with main Kobocog build
koboldcpp_opencog: $(OPENCOG_OBJECTS) koboldcpp_default
	g++ $(CXXFLAGS) $(OPENCOG_CXXFLAGS) $(OPENCOG_INCLUDES) \
		-o koboldcpp_opencog \
		ggml.o ggml-cpu.o ggml-ops.o ggml-vec.o ggml-binops.o ggml-unops.o \
		ggml_v3.o ggml_v2.o ggml_v1.o \
		expose.o gpttype_adapter.o model_adapter.o \
		$(OPENCOG_OBJECTS) \
		$(LDFLAGS)

# Documentation generation
.PHONY: docs-opencog
docs-opencog:
	@echo "Generating OpenCog-GGML documentation..."
	@echo "See opencog/README.md for detailed documentation"

# Validation checks
.PHONY: validate-opencog
validate-opencog:
	@echo "Validating OpenCog integration..."
	@echo "Checking header files..."
	@for header in $(OPENCOG_HEADERS); do \
		if [ -f "$$header" ]; then \
			echo "  ✓ $$header"; \
		else \
			echo "  ✗ $$header (missing)"; \
		fi; \
	done
	@echo "Checking source files..."
	@for source in $(OPENCOG_SOURCES); do \
		if [ -f "$$source" ]; then \
			echo "  ✓ $$source"; \
		else \
			echo "  ✗ $$source (missing)"; \
		fi; \
	done

# Help target
.PHONY: help-opencog
help-opencog:
	@echo "OpenCog-GGML Integration Targets:"
	@echo "  opencog-components  - Build all OpenCog components"
	@echo "  atomspace          - Build tensor AtomSpace component"
	@echo "  reasoning          - Build PLN reasoning engine"
	@echo "  attention          - Build ECAN attention system"
	@echo "  api               - Build cognitive API integration"
	@echo "  operations        - Build GGML custom operations"
	@echo "  test-opencog      - Run OpenCog integration tests"
	@echo "  clean-opencog     - Clean OpenCog object files"
	@echo "  validate-opencog  - Validate OpenCog file structure"
	@echo "  koboldcpp_opencog - Build Kobocog with OpenCog integration"
	@echo "  docs-opencog      - Show documentation information"

# Add OpenCog components to main build targets
koboldcpp_default: $(OPENCOG_OBJECTS)
	$(eval LDFLAGS += $(OPENCOG_OBJECTS))

# Integration check
.PHONY: check-integration
check-integration:
	@echo "Checking OpenCog-GGML integration status:"
	@echo "  AtomSpace: $(shell [ -f opencog/atomspace/tensor_atomspace.h ] && echo 'Ready' || echo 'Missing')"
	@echo "  PLN Engine: $(shell [ -f opencog/reasoning/pln_tensor_engine.h ] && echo 'Ready' || echo 'Missing')"
	@echo "  ECAN System: $(shell [ -f opencog/attention/ecan_attention.h ] && echo 'Ready' || echo 'Missing')"
	@echo "  API Integration: $(shell [ -f opencog/api/cognitive_api.h ] && echo 'Ready' || echo 'Missing')"
	@echo "  GGML Operations: $(shell [ -f opencog/operations/ggml_opencog_ops.h ] && echo 'Ready' || echo 'Missing')"
	@echo "  Test Suite: $(shell [ -f opencog/tests/test_tensor_atomspace.py ] && echo 'Ready' || echo 'Missing')"