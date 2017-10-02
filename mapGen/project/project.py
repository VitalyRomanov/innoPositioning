from mapGen.project.map_builder import MapBuilder


def main():
    builder = MapBuilder(file_name="walls.txt")
    builder.build_coverage_map(serialize=True)


main()

