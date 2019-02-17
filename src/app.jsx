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
        return (<div>{this.state.output}</div>)
    }
}

export default class App extends React.Component {
    render() {
        return (
            <div>
                <h2>Welcome to React!</h2>
                <h3>Output from Python:</h3>
                <RunPython />
            </div>
        );
    }
}
