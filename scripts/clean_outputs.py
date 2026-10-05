"""Remove somente os resultados gerados, preservando o .gitkeep."""

from pathlib import Path


def main() -> None:
    output_dir = Path("outputs").resolve()
    if output_dir.name != "outputs":
        raise RuntimeError("Diretorio de saida inesperado; limpeza cancelada.")
    for path in output_dir.iterdir():
        if path.name != ".gitkeep" and path.is_file():
            path.unlink()


if __name__ == "__main__":
    main()

