"""Resolve the frozen study's former root without altering its evidence."""
from contextlib import contextmanager
from pathlib import Path
import haiku_judge as judge

OLD_ROOT=Path('~/reports/courtier_sycophancy')
NEW_ROOT=Path(__file__).resolve().parent


def _path(value):
    path=Path(value)
    try:
        relative=path.relative_to(OLD_ROOT)
    except ValueError:
        return path
    if ".." in relative.parts:
        raise ValueError("invalid historical path")
    return NEW_ROOT/relative


@contextmanager
def paths():
    original_judge=judge.Path
    original_identity=judge.identity_check.Path
    judge.Path=_path
    judge.identity_check.Path=_path
    try:
        yield
    finally:
        judge.Path=original_judge
        judge.identity_check.Path=original_identity


def prepare(*args,**kwargs):
    with paths():return judge.prepare(*args,**kwargs)


def validate_completed(*args,**kwargs):
    with paths():return judge.validate_completed(*args,**kwargs)
