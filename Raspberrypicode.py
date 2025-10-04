import numpy as np
import pickle
import time
# Mock GPIO for testing on a non-Pi machine.
# On the Raspberry Pi, you would install and use the real RPi.GPIO library
try:
    import RPi.GPIO as GPIO
    print("Running on Raspberry Pi.")
except (ImportError, RuntimeError):
    print("Not on a Pi. Using a mock GPIO library.")
    import atexit
    class MockGPIO:
        def __init__(self):
            self.BCM = "BCM"
            self.OUT = "OUT"
            self.LOW = 0
            self.HIGH = 1
        def setmode(self, mode): print(f"GPIO Mode: {mode}")
        def setup(self, pin, mode): print(f"Setup Pin {pin} as {mode}")
        def output(self, pin, state): print(f"Set Pin {pin} to {'HIGH' if state else 'LOW'}")
        def cleanup(self): print("GPIO Cleanup.")
    GPIO = MockGPIO()
    atexit.register(GPIO.cleanup)


# --- 1. Hardware Pin Configuration ---
FAN_PIN = 17
PUMP_PIN = 18

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(FAN_PIN, GPIO.OUT)
GPIO.setup(PUMP_PIN, GPIO.OUT)

# --- 2. Sensor Reading Functions (PLACEHOLDERS) ---
# You will need to replace these with real sensor reading code.
def read_temperature_humidity():
    """Reads from a DHT22 sensor. Returns (temp_c, humidity_percent)."""
    # Placeholder: Replace with Adafruit_DHT library code
    temp = np.random.uniform(20, 30)
    humidity = np.random.uniform(65, 85)
    print(f"Reading [Mock Temp={temp:.1f}C, Mock Humidity={humidity:.1f}%]")
    return temp, humidity

def read_soil_moisture():
    """Reads from a soil moisture sensor. Returns percentage."""
    # Placeholder: Replace with code for your specific sensor (e.g., ADC reading)
    moisture = np.random.uniform(40, 75)
    print(f"Reading [Mock Soil Moisture={moisture:.1f}%]")
    return moisture

# --- 3. RL Agent and Environment Helper Functions ---
def load_q_table(filename='q_table_greenhouse.pkl'):
    """Loads the trained Q-table from a file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def discretize_state(state, low_bounds, high_bounds, num_buckets):
    """Converts a continuous state from sensors into a discrete one for the Q-table."""
    state_indices = []
    for i in range(len(state)):
        scaling = (state[i] - low_bounds[i]) / (high_bounds[i] - low_bounds[i])
        new_index = int(round((num_buckets[i] - 1) * scaling))
        new_index = max(0, min(new_index, num_buckets[i] - 1))
        state_indices.append(new_index)
    return tuple(state_indices)

# --- 4. Main Controller Loop ---
if __name__ == '__main__':
    print("Starting Smart Greenhouse Controller...")
    
    # Load the trained "brain"
    q_table = load_q_table()
    
    # These must match the values used in the training script
    low_bounds = np.array([15, 40, 20])
    high_bounds = np.array([35, 90, 80])
    num_buckets = (10, 10, 10)
    
    actions = {0: "Do Nothing", 1: "Water", 2: "Run Fan"}
    
    try:
        while True:
            # --- Step 1: Read the current state from sensors ---
            temp, humidity = read_temperature_humidity()
            soil_moisture = read_soil_moisture()
            continuous_state = np.array([temp, humidity, soil_moisture])
            
            # --- Step 2: Convert state to a Q-table index ---
            current_state_index = discretize_state(continuous_state, low_bounds, high_bounds, num_buckets)
            
            # --- Step 3: Choose the best action from the Q-table (no exploration) ---
            action_index = np.argmax(q_table[current_state_index])
            action_name = actions[action_index]
            print(f"Current State Index: {current_state_index}, Chosen Action: {action_name}")
            
            # --- Step 4: Execute the chosen action ---
            if action_name == "Water":
                GPIO.output(PUMP_PIN, GPIO.HIGH)
                time.sleep(5) # Run pump for 5 seconds
                GPIO.output(PUMP_PIN, GPIO.LOW)
            elif action_name == "Run Fan":
                GPIO.output(FAN_PIN, GPIO.HIGH)
                time.sleep(60) # Run fan for 1 minute
                GPIO.output(FAN_PIN, GPIO.LOW)

            # Wait for the next control cycle
            print("--- Cycle Complete. Waiting 1 hour. ---\n")
            time.sleep(3600) # Wait for an hour

    except KeyboardInterrupt:
        print("Program stopped by user.")
    finally:
        GPIO.cleanup()
        print("GPIO cleaned up. Exiting.")
