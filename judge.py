import subprocess
import tempfile
import os
import re



FUNC_BANNED_WORDS = ['implemented_by', 'noncomputable']


def extract_imports (code:str):
        imports=''
        rest=''
        for line in code.splitlines():
            if line.startswith('import'):
                imports+=line+'\n'
            else:
                rest+=line+'\n'
        return imports,rest


def check_lean_proof(challenge: str, submission: str) -> dict:
    for w in FUNC_BANNED_WORDS:
        if w in submission:
            return {
                "is_correct": False,
                "feedback": f"{w} is not allowed in the function implementation",
            }
    basedir=os.path.abspath('temp')
    print('basedir ',basedir)
    with tempfile.TemporaryDirectory(dir=basedir) as tmpdir:
        # Create temporary Lean files
        print('created temp directory ', tmpdir)
        

        targf=os.path.join(tmpdir, "target.lean")

        with open(targf, "w") as f:
            f.write(challenge)
        print('finished writing to '+targf)

        prooff=os.path.join(tmpdir, "proof.lean")

        with open(prooff, "w") as f:
            f.write(submission)

        def compile (fname):
            # compile on the temporary file
            assert fname.endswith('.lean')
            ofname=fname[:-4]+'olean'
            result = subprocess.run(["lake","env","lean",'-o',ofname, fname], capture_output=True, text=True)
            # Check if Lean 4 succeeded (return code 0 means success)
            is_correct = result.returncode == 0
            return is_correct, result.stderr + result.stdout

        def compare(targf, subf):
            for f in [targf,subf]:
              r,err=compile(f)
              print (f"finished compiling {f}")
              if not r:
                err=f"Compilation error for {f}:\n"+err
                return r,err
            otarg=targf[:-4]+'olean'
            osub=subf[:-4]+'olean'
            result=subprocess.run(["lake","env","safe_verify",otarg,osub],capture_output=True,text=True)
            is_correct = result.returncode==0
            return is_correct, result.stderr+result.stdout

        is_correct, error_message = compare(targf, prooff)
        
        return {
            "is_correct": is_correct,
            "feedback": error_message.strip() if error_message else "Proof checked successfully!",
        }
