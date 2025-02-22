// Global
const board = []
const SIZE = 9

let show_policy = true
let show_value = true

// Elts
const eval_bar = document.getElementById("eval-bar")
const eval_num = document.getElementById("eval-num")

const board_elt = document.getElementById("board")

const nav_reset = document.getElementById("reset")
const nav_undo = document.getElementById("undo")
const nav_pass = document.getElementById("pass")

const settings_policy = document.getElementById("show-policy")
const settings_value = document.getElementById("show-value")

// Functions
async function updateBoard() {
    const response1 = await fetch("/get_board", {"method": "POST"})
    const newBoardStr = await response1.text()

    const temp = newBoardStr.split("\n")

    for (let i = 0; i < SIZE; i++) {
        for (let j = 0; j < SIZE; j++) {
            board[i][j].innerText = temp[i][j]
        }
    }

    const response2 = await fetch("/network", {"method": "POST"})
    const things = await response2.json()

    const policy = things["policy"]
    const value = things["value"]

    updatePolicy(policy)
    updateEval(value)
}

function updateEval(value) {
    eval_bar.setAttribute("style", "--eval: " + value)
    eval_num.innerText = Math.round(value * 100) / 100

    if (value >= 0) {
        eval_num.classList.remove("white")
        eval_num.classList.add("black")
    } else {
        eval_num.classList.add("white")
        eval_num.classList.remove("black")
    }
}

function updatePolicy(policy) {
    for (let i = 0; i < SIZE; i++) {
        for (let j = 0; j < SIZE; j++) {
            board[i][j].setAttribute("style", "--policy: " + policy[i * SIZE + j])
        }
    }
}

async function playMove(i, j) {
    await fetch("/play_move", {
        "method": "POST",
        "headers": {
            "Content-Type": "application/json",
        },
        "body": JSON.stringify({
            "row": i,
            "col": j
        })
    })

    updateBoard()
}

async function reset() {
    await fetch("/reset", {"method": "POST"})

    updateBoard()
}

async function undo() {
    await fetch("/undo", {"method": "POST"})

    updateBoard()
}

async function pass() {
    playMove(-1, -1)
}

function togglePolicy() {
    show_policy = !show_policy

    if (show_policy) {
        board_elt.classList.add("show-policy")
    } else {
        board_elt.classList.remove("show-policy")
    }
}

function toggleValue() {
    show_value = !show_value

    if (show_value) {
        eval_bar.classList.remove("hidden")
    } else {
        eval_bar.classList.add("hidden")
    }
}

// Main
nav_reset.addEventListener("click", reset)
nav_undo.addEventListener("click", undo)
nav_pass.addEventListener("click", pass)

settings_policy.addEventListener("click", togglePolicy)
settings_value.addEventListener("click", toggleValue)

for (let i = 0; i < SIZE; i++) {
    const row = []
    const rowElt = document.createElement("div")
    rowElt.classList.add("row")

    for (let j = 0; j < SIZE; j++) {
        const temp = document.createElement("button")

        temp.addEventListener("click", () => {playMove(i, j)})
        row.push(temp)

        rowElt.appendChild(temp)
    }

    board.push(row)
    board_elt.appendChild(rowElt)
}

updateBoard()
