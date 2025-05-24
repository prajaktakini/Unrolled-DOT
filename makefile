.PHONY: all create-env install-deps activate-note init-conda-shell

ENV_NAME := unrolled-dot-env
PYTHON_VERSION := 3.10
CONDA_DIR := $(HOME)/miniconda3
CONDA_BIN := $(CONDA_DIR)/bin/conda

all: create-env init-conda-shell install-deps activate-note

# Step 1: Install Miniconda if not present
$(CONDA_BIN):
	@echo "🔍 Conda not found. Installing Miniconda..."
	@wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
	@bash miniconda.sh -b -p $(CONDA_DIR)
	@rm miniconda.sh
	@echo "✅ Miniconda installed."

# Step 2: Create Conda environment
create-env: $(CONDA_BIN)
	@echo "🔍 Checking if Conda environment '$(ENV_NAME)' exists..."
	@if $(CONDA_BIN) env list | grep -q $(ENV_NAME); then \
		echo "✅ Environment '$(ENV_NAME)' already exists."; \
	else \
		echo "🛠️  Creating Conda environment '$(ENV_NAME)' with Python $(PYTHON_VERSION)..."; \
		$(CONDA_BIN) create -y -n $(ENV_NAME) python=$(PYTHON_VERSION); \
	fi

# Step 3: Initialize Conda shell integration
init-conda-shell:
	@echo "🔧 Initializing Conda shell support..."
	@$(CONDA_BIN) init bash
	@echo "🔄 Sourcing ~/.bashrc to enable 'conda' command..."
	@bash -c "source ~/.bashrc && echo '✅ Conda shell initialized.'"

# Step 4: Install GPU-supported dependencies
# Step 4: Install GPU-supported dependencies
install-deps:
	@echo "📦 Installing Python and GPU dependencies into '$(ENV_NAME)'..."
	@$(CONDA_BIN) install -n $(ENV_NAME) -y \
		pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
	@$(CONDA_BIN) run --no-capture-output -n $(ENV_NAME) pip install \
		numpy>=1.26 matplotlib>=3.8 kornia==0.5.11 \
		parse scipy scikit-image h5py jupyterlab huggingface_hub ipywidgets

activate-note:
	@echo ""
	@echo "✅ Setup complete!"
	@echo "👉 To activate the environment manually, run:"
	@echo ""
	@echo "   conda activate $(ENV_NAME)"
	@echo ""