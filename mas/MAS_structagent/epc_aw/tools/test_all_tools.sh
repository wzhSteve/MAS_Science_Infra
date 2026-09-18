#!/bin/bash

# find all tool.py files in the tools folder
tools=$(find . -type f -name "tool.py")

echo "Testing all tools"
echo "Tools:"
for tool in $tools; do
    echo "  - $(basename $(dirname $tool))"
done

echo ""
echo "Running tests in parallel..."

# Export function so parallel can use it
run_test() {
    local tool="$1"
    local tool_dir=$(dirname "$tool")
    local tool_name=$(basename "$tool_dir")

    echo "Testing $tool_name..."

    pushd "$tool_dir" > /dev/null || exit 1

    python tool.py > test.log 2>&1
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "❌ $tool_name failed! Check $tool_dir/test.log for details" >&2
        exit $exit_code
    else
        echo "✅ $tool_name passed"
    fi

    popd > /dev/null
}
export -f run_test

export tools
# Run all tests in parallel, max 8 at a time
echo "$tools" | tr ' ' '\n' | parallel -j 8 run_test

# Capture overall success/failure
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tests passed"
    exit 0
else
    echo ""
    echo "❌ Some tests failed"
    exit 1
fi