import React from 'react';

class RunPython extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            output: "Awaiting output from Python ... ",
        };
        
        // SOURCE: https://www.techiediaries.com/python-electron-tutorial/
        var python = require('child_process').spawn('python3', ['./src/dummy.py']);
        python.stdout.on('data', (data) =>  {
            this.setState({
                output: data.toString('utf8'),
            });
        });
    }

    render() {
        return (<div>{this.state.output}</div>);
    }
}

export default class App extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            paths: null,
        }
    }

    selectFiles() {
        const { dialog } = require('electron').remote;
        let selected_files = dialog.showOpenDialog({
            properties: ["openFile"],
            title: "Select Video",
            defaultPath: "~",
            buttonLabel: "Analyze",
            filters: [
                { name: "Movies", extensions: ["avi", "mp4"]},
                { name: "TESTING", extensions: ["*"]},
            ],
        });
        this.setState({
            paths: selected_files,
        });
        return selected_files;
    }

    analyze() {
        const files = this.state.paths;
        if (!files) {
            alert("You must select a file to analyze.");
            return;
        }
        alert("TODO: Analyze these files: " + files.toString("utf8"));
    }

    render() {
        let files = "No Files Selected";
        
        if (this.state.paths) { 
            files = this.state.paths.map((path) => {
                return(
                    <li key={path}>{path}</li>
                );
            });
        }
        
        return (
            <div>
                <h2>Welcome to React!</h2>
                
                <h3>Output from Python:</h3>
                <RunPython />
                <button onClick={() => this.analyze()}>Analyze Files</button> 
                <h3>Selected Files:</h3>
                <button onClick={() => this.selectFiles()}>Select Files</button>
                <div>{files}</div>
            </div>
        );
    }
}
