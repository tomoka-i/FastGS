{
  description = "FastGS developer tooling shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        python = pkgs.python3;
      in
      {
        devShells.default = pkgs.mkShell {
          name = "fastgs-dev";

          packages =
            [
              pkgs.bashInteractive
              pkgs.cargo
              pkgs.clippy
              pkgs.git
              pkgs.pkg-config
              pkgs.rustc
              pkgs.rustfmt
              python
              pkgs.maturin
              python.pkgs.pip
              python.pkgs.setuptools
              python.pkgs.wheel
            ]
            ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
              pkgs.libiconv
            ];

          env = {
            FASTGS_NIX_MODE = "tooling-only";
          };

          shellHook = ''
            echo "FastGS Nix shell: tooling-only mode"
            echo "Python: $(python --version)"
            echo "Rust: $(rustc --version)"
            if [ -d csrc_rust ]; then
              echo "Rust extension checks:"
              echo "  cd csrc_rust"
              echo "  cargo fmt --check"
              echo "  cargo check"
              echo "  maturin develop --release"
            else
              echo "csrc_rust is not present on this branch; merge the Rust branch to run extension checks."
            fi
          '';
        };
      }
    );
}
