"""Catch runtime annotation/import errors before a P4 change reaches integration."""

import importlib
import pkgutil

import p4_eval
import p4_pipeline
import p4_stats
import tbcore


def test_p4_and_shared_modules_import():
    for package in (p4_stats, p4_pipeline, p4_eval, tbcore):
        for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            importlib.import_module(module.name)
