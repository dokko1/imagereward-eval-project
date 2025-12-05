FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

# Install system dependencies required for dbus-python and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    zsh \
    vim \ 
    pip \
    build-essential \
    libdbus-1-dev \
    libffi-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /root/ImageReward/

# Copy the cog.yaml and requirements.txt
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Install Powerlevel10k theme
RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/themes/powerlevel10k && \
    sed -i 's/^ZSH_THEME=".*"/ZSH_THEME="powerlevel10k\/powerlevel10k"/' ~/.zshrc

# Install plugins
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && \
    sed -i '/^plugins=/s/(/(git zsh-syntax-highlighting zsh-autosuggestions /' ~/.zshrc

RUN chsh -s /bin/zsh

