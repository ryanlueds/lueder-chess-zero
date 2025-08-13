import { Chess } from "chess.js";
import { Chessground } from "chessground";
import type { Api } from "chessground/api";
import type { Key } from "chessground/types";
import "./style.css";
import "chessground/assets/chessground.base.css";
import "chessground/assets/chessground.brown.css";
import "chessground/assets/chessground.cburnett.css";

const chess = new Chess();
const fenDisplay = document.getElementById("fen-display") as HTMLDivElement;

const boardElement = document.getElementById("board") as HTMLElement;
let ground: Api;

function updateFenDisplay() {
    fenDisplay.textContent = chess.fen();
}

function toDests(chess: Chess) {
    const dests = new Map<Key, Key[]>();
    chess.moves({ verbose: true }).forEach(m => {
        const from = m.from as Key;
        const to = m.to as Key;
        const existing = dests.get(from) || [];
        dests.set(from, [...existing, to]);
    });
    return dests;
}

async function getComputerMove() {
    const fen = chess.fen();
    try {
        const response = await fetch('http://localhost:5000/get_move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ fen: fen, simulations: 2000 }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, ${errorText}`);
        }

        const data = await response.json();
        const moveStr = data.move;

        if (moveStr) {
            const from = moveStr.substring(0, 2);
            const to = moveStr.substring(2, 4);
            const promotion = moveStr.length > 4 ? moveStr.substring(4, 5) : undefined;
            
            chess.move({ from, to, promotion });
            
            ground.set({ 
                fen: chess.fen(),
                turnColor: 'white',
                movable: {
                    color: 'white',
                    dests: toDests(chess)
                }
            });
            updateFenDisplay();
        }
    } catch (error) {
        console.error("Failed to get computer move:", error);
    }
}

function initBoard() {
    ground = Chessground(boardElement, {
        fen: chess.fen(),
        orientation: "white",
        turnColor: 'white',
        movable: {
            color: 'white',
            free: false,
            dests: toDests(chess),
        },
        events: {
            move: (orig, dest) => {
                const move = chess.move({ from: orig, to: dest, promotion: "q" });
                if (move) {
                    ground.set({ 
                        fen: chess.fen(),
                    });
                    updateFenDisplay();
                    
                    if (chess.turn() === 'b') {
                        ground.set({
                            turnColor: 'black',
                            movable: {
                                color: 'black',
                                dests: toDests(chess)
                            }
                        });
                        setTimeout(getComputerMove, 100);
                    }
                } else {
                    ground.set({ fen: chess.fen() });
                }
            },
        },
    });
    updateFenDisplay();
}

initBoard(); 
