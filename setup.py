from __future__ import annotations

from setuptools import Extension, setup


setup(
    ext_modules=[
        Extension(
            "kalah_zero._fast_game",
            sources=["src/kalah_zero/_fast_game.cpp"],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17"],
        )
    ],
)
