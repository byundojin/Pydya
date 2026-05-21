"""README 예시로 Pydya 동작을 보여준다.

실행:  python examples/readme_example.py
"""

from pydya import compile_source

SOURCE = """\
V = CompileVar[int]()

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
