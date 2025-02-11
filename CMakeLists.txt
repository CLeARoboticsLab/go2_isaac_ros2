cmake_minimum_required(VERSION 3.5)
project(go2_isaac_ros2)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install Python scripts
ament_python_install_package(${PROJECT_NAME})

# Install Python entry points
install(PROGRAMS
  go2_isaac_ros2/main.py
  DESTINATION lib/${PROJECT_NAME}
)


ament_package()
