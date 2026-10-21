FROM ubuntu:noble

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl \
      wget \
      git \
      openssh-client \
      sudo \
      just

RUN usermod -aG sudo "ubuntu" && \
    echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/ubuntu" && \
    chmod 440 "/etc/sudoers.d/ubuntu" && \
    git config --system core.sshCommand /usr/bin/ssh

USER ubuntu
WORKDIR /home/ubuntu/workspace

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
