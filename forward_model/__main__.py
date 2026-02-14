import typer

app = typer.Typer()


@app.command("run")
def run() -> None:
    print("Hello World!")


if __name__ == "__main__":
    app()
