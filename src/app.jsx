import React from 'react';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Button from 'react-bootstrap/Button';
import ListGroup from 'react-bootstrap/ListGroup';

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
                { name: "TESTING", extensions: ["*"]},
                { name: "Movies", extensions: ["avi", "mp4"]},
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
                    <ListGroup.Item>{path}</ListGroup.Item>
                );
            });
        }
        
        return (
            <div>
            <link
              rel="stylesheet"
              href="https://maxcdn.bootstrapcdn.com/bootstrap/4.2.1/css/bootstrap.min.css"
              integrity="sha384-GJzZqFGwb1QTTN6wy59ffF1BuGJpLSa9DkKMp0DgiMDm4iYMj70gZWKYbI706tWS"
              crossOrigin="anonymous"
            />
            <Container>
                <h3>Output from Python:</h3>
                <RunPython />
                <Row className="justify-content-md-center">
                    <Col>
                        <Button variant="primary" onClick={() => 
                            this.analyze()}>Analyze Files</Button> 
                    </Col>
                    <Col>
                        <Button variant="primary" onClick={() => 
                            this.selectFiles()}>Select Files</Button>
                    </Col>
                </Row>
            </Container>
                <h3>Selected Files:</h3>
                <ListGroup>{files}</ListGroup>
            </div>
        );
    }
}
