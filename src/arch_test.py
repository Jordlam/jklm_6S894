import platform

def test_arch():
    print("We are executing on", platform.uname()[4])

if __name__ == "__main__":
    test_arch()
