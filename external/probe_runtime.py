"""Process-local workaround for restricted-runner Windows mkdir(0700) ACLs.

No change to Perseus sources or machine settings; all temporary paths stay in external/.
"""
from pathlib import Path
import tempfile
import uuid

def configure_tempdirs():
    external=Path(__file__).resolve().parent
    tmp=external/'tmp'
    tmp.mkdir(exist_ok=True)
    def mkdtemp(suffix=None,prefix=None,dir=None):
        parent=Path(dir or tmp).resolve()
        if not parent.is_relative_to(external):
            raise ValueError('Temporary output outside external is forbidden')
        path=parent/f'{prefix or "tmp"}{uuid.uuid4().hex}{suffix or ""}'
        path.mkdir(mode=0o777)
        return str(path)
    tempfile.mkdtemp=mkdtemp
    tempfile.tempdir=str(tmp)
