package main

import (
	"fmt"
	"os"

	"github.com/mechanical-lich/brain-lib/pkg/brain/v3"
	"github.com/mechanical-lich/brain-lib/pkg/brain/v3/validation"
)

func main() {
	args := os.Args[1:]
	if len(args) != 2 {
		fmt.Println("Pass a brain json to validate")
		return
	}

	cmd := args[0]

	brainPath := args[1]
	b, err := brain.LoadBrain(brainPath)

	if err != nil {
		fmt.Printf("Error loading brain: %v\n", err)
		return
	}

	switch cmd {
	case "validate":
		complaints := validation.ValidateBrain(b)
		if len(complaints) > 0 {
			fmt.Println("Validation Failed:")
			for i, c := range complaints {
				fmt.Printf("[%d]: %v (id=%s)\n", i, c.ComplaintType, c.ID)
			}
			return
		}

		fmt.Println("Brain validation succeeded")
	case "repair":
		complaints := validation.ValidateBrain(b)
		if len(complaints) == 0 {
			fmt.Println("No complaints found; nothing to repair")
		} else {
			err = validation.FixSynapses(b, complaints)
			if err != nil {
				fmt.Printf("Error fixing synapses: %v\n", err)
				return
			}
			err = b.Save(brainPath + ".fixed")
			if err != nil {
				fmt.Printf("Error saving brain: %v\n", err)
				return
			}
			fmt.Println("Brain repair succeeded")
		}
	default:
		fmt.Println("Unknown command:", cmd)
	}

}
