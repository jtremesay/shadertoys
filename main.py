import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl

    return (pl,)


@app.cell
def _(pl):
    df = pl.read_parquet("bad_apple/video_pixels.parquet")
    df
    return


if __name__ == "__main__":
    app.run()
