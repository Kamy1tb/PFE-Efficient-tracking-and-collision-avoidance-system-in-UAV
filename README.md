# PFE-Efficient-tracking-and-collision-avoidance-system-in-UAV

## Description
This repository encompasses a project developed within the framework of the End-of-Study project at the National School of Computer Science of Algiers. The project is centered around simulating the behavior of a drone in a dynamic environment. The primary objective is to utilize advanced technologies to enable the drone to perform various tasks autonomously.

The core functionality of the project revolves around the drone's ability to identify and track a designated target using object detection techniques. This involves employing sophisticated algorithms to process visual data in real-time, enabling the drone to recognize and lock onto the target accurately.

To achieve this, the project leverages state-of-the-art reinforcement learning methodologies. Through continuous interaction with the environment, the drone learns optimal strategies for tracking the target efficiently while navigating through complex scenarios. The reinforcement learning algorithms enable the drone to adapt its behavior based on feedback received from its surroundings, thereby enhancing its overall performance.

A key aspect of the project is the integration of obstacle avoidance capabilities into the drone's navigation system. This involves developing algorithms that enable the drone to detect and circumvent both static and dynamic obstacles in its path. By incorporating obstacle avoidance mechanisms, the drone can operate safely in diverse environments without the risk of collisions or disruptions.

Furthermore, the project focuses on optimizing the drone's path planning process to ensure efficiency and cost-effectiveness. By analyzing various factors such as distance, terrain, and energy consumption, the project aims to generate optimal trajectories that minimize resource usage while achieving the desired objectives.

The simulation environment for the project utilizes the Airsim simulator, which is powered by Unreal Engine. This combination provides a realistic and immersive platform for testing and validating the drone's capabilities. Additionally, the project utilizes the Airsim API to facilitate integration with Python, enabling seamless development and implementation of control algorithms.

## Installation

### Step 1: Epic Games Launcher
- Install the Epic Games Launcher from [this link](https://store.epicgames.com/en-US/download/en-US/).
- Install Unreal Engine version 4.27.2 through the Epic Games Launcher.

### Step 2: Clone AirSim Repository
- Clone the AirSim GitHub repository locally using the following command:
git clone https://github.com/microsoft/AirSim.git

### Step 3: Install Visual Studio 2022
- Download and install Visual Studio 2022 from [this link](https://visualstudio.microsoft.com/vs/community/).
- Within the Visual Studio Installer, make sure to install:
- Desktop Development with C++
- Windows 11 SDK
- .NET SDK

### Step 4: Build AirSim
- Open Visual Studio 2022 and launch the Developer Command Prompt for VS 2022.
- Navigate to the AirSim directory and execute the `build.cmd` command.
This will generate the `Unreal\Plugins` directory within the AirSim directory.

### Step 5: Generate Visual Studio Project Files
- Open your project's `.exe` file in Visual Studio 2022.
- Right-click and select "Generate Visual Studio Project Files."

### Step 6: Configure Unreal Project
- Open your Unreal project.
- Switch the game mode to `AirsimGamemode`.

### Step 7: Configure Settings
- Locate the `settings.json` file in your AirSim folder.
- Replace its contents with the contents provided in this repository.

### Step 8: Launch AirSim Python API
- Update the IP addresses in both the `settings.json` file and the `client.py` file available in the `PythonClient` folder.

Now you're ready to utilize the AirSim Python API seamlessly.
