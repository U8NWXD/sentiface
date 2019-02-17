import React from 'react';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Button from 'react-bootstrap/Button';
import ListGroup from 'react-bootstrap/ListGroup';
import Modal from 'react-bootstrap/Modal';
import ButtonToolbar from 'react-bootstrap/ButtonToolbar';
import Badge from 'react-bootstrap/Badge';
import * as d3 from 'd3';
import {XYPlot, VerticalBarSeries, XAxis, YAxis} from 'react-vis';

/*
 * props:
 * * show: boolean for whether to display modal
 * * onHide: callback to execute when modal closed
 * * body: text to display as the body
 */
function ExecutingPythonModal(props) {
    return (
        <Modal
            show={props.show}
            // TODO: Remove the line below?
            //onHide={props.onHide}
            size="lg"
            aria-labelledby="contained-modal-title-vcenter"
            centered
        >
            <Modal.Header closeButton>
                <Modal.Title id="contained-modal-title-vcenter">
                    Analysis Results
                </Modal.Title>
            </Modal.Header>
            <Modal.Body>
                {props.body}
            </Modal.Body>
            <Modal.Footer>
                <Button 
                    variant="primary" onClick={props.onHide}
                >Close</Button>
            </Modal.Footer>
        </Modal>
    );
}

class RunPython extends React.Component {
    /*
     * props:
     * * python_args: Array of arguments to pass to the python3 call
     */
    constructor(props) {
        super(props);

        this.state = {
            output: "Awaiting output from Python ... ",
            modalShow: false,
            python_args: props.python_args
        };
    }

    executePython() {
        // SOURCE: https://www.techiediaries.com/python-electron-tutorial/
        var python = require('child_process').spawn(
            'python3', this.state.python_args);
        python.stdout.on('data', (data) =>  {
            this.setState({
                output: data.toString('utf8'),
            });
        });
    }

    handleClick() {
        this.setState({ modalShow: true });
        this.executePython();
    }

    render() {
        let modalClose = () => this.setState({ modalShow: false })
        return (
            <ButtonToolbar>
            <Button
                variant="primary"
                onClick={() => this.handleClick()}
            >Analyze</Button>

            <ExecutingPythonModal
                show={this.state.modalShow}
                onHide={modalClose}
                body={this.state.output}
            />
            </ButtonToolbar>
        );
    }
}

function BarPlot(props) {
    console.log(props);
    const data = props.data
    const height = props.height ? props.height : 300;
    const width = props.width ? props.width : 300;
    // SOURCE: https://stackoverflow.com/a/10050831
    const x_labels = props.x_labels ? props.x_labels : [...Array(data.length).keys()];
    const margin = {left: 40, right: 10, top: 10, bottom: 100}
    return(
        <div>
            <XYPlot height={height} width={width} margin={margin}>
                <VerticalBarSeries data={data} />
                <XAxis tickFormat={i => x_labels[i]} tickLabelAngle={-90}/>
            </XYPlot>
        </div>
    );
}

class VisualizeData extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            data: null,
            path: null,
        };
        this.state.path = "../sotu.csv";
    }

    loadData(path) {
        if (!path) {
            alert("No data file selected.");
        }
        d3.csv(path).then((loaded) => this.setState({data: loaded}));
    }

    render() {
        if (this.state.data) {
            let emotions = ["anger", "contempt", "disgust", "fear", "happiness",
                "sadness", "smile", "surprise"];
            let toDisplay = [];
            let sumCallback = (acc, cur) => (acc + cur);

            for (let i = 0; i < emotions.length; i ++) {
                toDisplay[i] = {
                    x: i,
                    // SOURCE: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/Reduce
                    y: this.state.data.map(elem => 
                        100 * parseFloat(elem[emotions[i]]))
                        .reduce(sumCallback, 0.0),
                }
            }
            console.log(toDisplay);
            return (
                <BarPlot data={toDisplay} height={400} width={400} x_labels={emotions}/>
            );
        } else {
            return ( 
                <Button variant="primary" onClick={() => 
                    this.loadData(this.state.path)}>Load Data</Button>
            );
        }
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
                    <ListGroup.Item key={path}>{path}</ListGroup.Item>
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
                <Row className="justify-content-md-center">
                    <h2>
                        <Badge variant="secondary">
                            Sentiface Sentiment Analyzer
                        </Badge>
                    </h2>
                </Row>
                <Row className="justify-content-md-center">
                    <Col>
                        <RunPython python_args={
                            ['./src/dummy.py'].concat(this.state.paths)
                        } />
                    </Col>
                    <Col>
                        <Button variant="primary" onClick={() => 
                            this.selectFiles()}>Select Files</Button>
                    </Col>
                </Row>
                <Row>
                    <h3>Selected Files:</h3>
                </Row>
                <Row>
                    <ListGroup>{files}</ListGroup>
                </Row>
                <Row>
                    <VisualizeData />
                </Row>
            </Container>
            </div>
        );
    }
}
