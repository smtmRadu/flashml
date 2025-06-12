import os


DOTNET_TARGET_FRAMEWORK = "net8.0"


def run_compiled_csharp(executable_path, arguments=None):
    import subprocess

    """
    Runs a compiled C# executable and captures its output.
    This is a helper function, similar to the original run_csharp_script.

    Args:
        executable_path (str): The full path to the C# executable.
        arguments (list, optional): A list of string arguments to pass to the C# script.

    Returns:
        tuple: (stdout, stderr, return_code)
    """
    if not os.path.exists(executable_path):
        print(f"Error: C# executable not found at {executable_path}")
        return None, f"Executable not found at {executable_path}", -1

    command = [executable_path]
    if arguments:
        command.extend(arguments)

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        return process.stdout, process.stderr, process.returncode
    except FileNotFoundError:
        print(f"Error: Command '{executable_path}' not found during execution.")
        return None, f"Command '{executable_path}' not found.", -1
    except Exception as e:
        print(f"An error occurred while running the C# executable: {e}")
        return None, str(e), -1


def call_cs_kernel(cs_file_path: str, input_args: list = None) -> str:
    import tempfile
    import shutil
    import platform
    import subprocess

    """
    Compiles a C# source file and runs it, returning its standard output.

    Args:
        cs_file_path (str): Path to the C# source file (e.g., "MyCSharpKernel.cs").
        input_args (list, optional): A list of string arguments to pass to the C# executable.

    Returns:
        str: The standard output from the C# executable.
             Returns an error message string if compilation or execution fails.

    Raises:
        FileNotFoundError: If the cs_file_path does not exist.
        Exception: For compilation or runtime errors.
    """
    if not os.path.exists(cs_file_path):
        raise FileNotFoundError(f"C# source file not found: {cs_file_path}")

    # check input args are strings
    if input_args is not None:
        for arg in input_args:
            if not isinstance(arg, str):
                raise TypeError("C# kernel failed: Input arguments must be strings.")

    if input_args is None:
        input_args = []

    with tempfile.TemporaryDirectory() as temp_dir:
        base_cs_file_name = os.path.basename(cs_file_path)
        kernel_name = os.path.splitext(base_cs_file_name)[0]

        temp_cs_path = os.path.join(
            temp_dir, base_cs_file_name
        )  # Use original filename
        shutil.copy(cs_file_path, temp_cs_path)

        csproj_content = f"""<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>{DOTNET_TARGET_FRAMEWORK}</TargetFramework>
    <AssemblyName>{kernel_name}</AssemblyName>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
    <LangVersion>latest</LangVersion> </PropertyGroup>
  <ItemGroup>
    </ItemGroup>
</Project>
"""
        csproj_path = os.path.join(temp_dir, f"{kernel_name}.csproj")
        with open(csproj_path, "w", encoding="utf-8") as f:
            f.write(csproj_content)

        publish_dir = os.path.join(temp_dir, "publish_output")

        compile_command = [
            "dotnet",
            "build",
            csproj_path,
            "-c",
            "Release",
            "-o",
            publish_dir,
        ]

        print(f"\033[94mCompiling C# code: {' '.join(compile_command)}\u001b[0m")
        compile_process = subprocess.run(
            compile_command,
            capture_output=True,
            text=True,
            cwd=temp_dir,
        )

        if compile_process.returncode != 0:
            error_message = (
                f"\033[91mC# compilation failed with exit code {compile_process.returncode}.\u001b[0m\n"
                f"\033[91mStdout:\n{compile_process.stdout}\u001b[0m\n"
                f"\033[91mStderr:\n{compile_process.stderr}\u001b[0m"
            )
            print(error_message)
            return error_message

        executable_name = kernel_name
        if platform.system() == "Windows":
            executable_name += ".exe"

        compiled_exe_path = os.path.join(publish_dir, executable_name)

        if not os.path.exists(compiled_exe_path):
            error_message = (
                f"\033[91mCompiled C# executable not found at expected path: {compiled_exe_path}\u001b[0m\n"
                f"\033[91mBuild stdout:\n{compile_process.stdout}\u001b[0m\n"
                f"\033[91mBuild stderr:\n{compile_process.stderr}\u001b[0m"
            )
            print(error_message)
            return error_message

        print(
            f"\033[94mRunning compiled C# executable: {compiled_exe_path} with {len(input_args)} args.\u001b[0m"
        )
        stdout, stderr, return_code = run_compiled_csharp(compiled_exe_path, input_args)

        if return_code != 0:
            error_message = (
                f"\033[91mC# executable '{compiled_exe_path}' failed with exit code {return_code}.\u001b[0m\n"
                f"\033[91mStdout:\n{stdout}\u001b[0m\n"
                f"\033[91mStderr:\n{stderr}\u001b[0m"
            )
            print(error_message)
            return error_message

        return stdout.strip()
