"""Demonstrate Pydya on the README example.

Run with:  python examples/readme_example.py
"""

from pydya import compile_source

SOURCE = """\
V = CompileVar('V')

def f(a):
    return a + V

if V < 5:
    a = f(5)
else:
    a = 5

b = a + V

print(a)
print(b)
"""

if __name__ == "__main__":
    compiled = compile_source(SOURCE, env={"V": 3})
    print("=== compiled (V = 3) ===")
    print(compiled)
    print("=== running compiled output ===")
    exec(compiled, {})
