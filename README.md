## TetroMate: Your Google Assistant for Tetrode Lowering 

TetroMate is a Google Assistant agent designed to simplify and automate the process of documenting tetrode movements during electrophysiology experiments. It allows researchers to:

* **Log adjustments:** Record tetrode movements (up/down, number of turns) using natural language voice commands.
* **Track cell activity:** Note observations about cell activity (e.g., ripples, theta waves, cell count).
* **Mark experimental stages:** Define markers for different experimental sessions and epochs.
* **Set brain area:** Specify the brain area where each tetrode is currently located.
* **Track depth:** Query the current depth of a tetrode in turns and millimeters.
* **Generate summaries:** Automatically create organized summaries of lowering sessions, including a visually appealing depth chart.
* **Store data securely:** All data is directly saved to a Google Sheet, ensuring data integrity and accessibility.

**Benefits:**

* **Hands-free logging:**  Focus on the experiment while documenting data using voice commands.
* **Reduced errors:** Minimize manual data entry mistakes and inconsistencies.
* **Organized data:**  Easily access and analyze lowering session data. 
* **Time-saving:** Streamline the documentation process and free up time for other tasks.

## Getting Started

1. **Set up a Google Cloud Platform Project:** 
    * Create a new project on Google Cloud Platform ([https://cloud.google.com/](https://cloud.google.com/)). 
    * Enable the Google Sheets API.
    * Create a service account and download its JSON key file.

2. **Create a Google Sheet:**
    * Create a new Google Sheet to store your experimental data.
    * Add worksheets named "Raw," "Summary," "Mapping", and "Prediction". The "Prediction" sheet can be used to store prior experiment statistics.

3. **Configure TetroMate:**
    *  Open `tetromate_webserver.py` and modify the following:
        * `path`: Point to the path of your downloaded service account JSON key file.
        * `url_configuration_file`: Path to a file containing the URL of your Google Sheet.
        * `screw_type`: Specify the type of screw drive being used ("openefizz," "roshan," or "aught80").
        * `const_depth_mm`: Set a constant depth offset (in millimeters) if necessary.
        * `continuous_explode`: Adjust for whether your tetrode drive can adjust multiple tetrodes simultaneously.

4. **Deploy TetroMate:**
    * Use a tool like ngrok ([https://ngrok.com/](https://ngrok.com/)) to expose your local Flask server to the internet.
    * Update the `webhook` URL in `agent.json` with your ngrok URL.

5. **Create a Dialogflow Agent:**
    * Create a new Dialogflow agent ([https://dialogflow.cloud.google.com/](https://dialogflow.cloud.google.com/)).
    * Import the `agent.json` file into your Dialogflow agent.
    * Configure the fulfillment to use your ngrok webhook URL.

6. **Integrate with Google Assistant:**
    * Connect your Dialogflow agent to Google Assistant.
    * Test your agent through the Google Assistant simulator or on a Google Assistant-enabled device.

## Usage

Once deployed, you can use natural language commands to interact with TetroMate through Google Assistant:

* **Adjust tetrode:** "Lower 3 turns on tetrode 5." 
* **Record ripples:**  "Ripples are 2."
* **Set area:** "Area is CA1 for tetrodes 1, 2, and 3."
* **Get depth:** "What is the depth of tetrode 7?"
* **Create summary:** "Generate a pretty table." 
* **Backup data:** "Backup my data."
* **Undo last entry:** "Undo."
* **Add notes:** "Note: strong cell activity observed."
* **Set marker:** "Marker: Session 2."

## Contributing

Contributions to TetroMate are welcome! Please feel free to submit issues or pull requests on this repository. 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
