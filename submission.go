package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
)

type DataPoint struct {
	Features []float64
	Label    string
}

func LoadDataset(filename string) ([]DataPoint, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var dataset []DataPoint
	for _, record := range records {
		var features []float64
		for _, value := range record[:len(record)-1] {
			feature, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return nil, err
			}
			features = append(features, feature)
		}
		label := record[len(record)-1]
		dataset = append(dataset, DataPoint{Features: features, Label: label})
	}

	return dataset, nil
}

func EuclideanDistance(point1, point2 []float64) float64 {
	var sum float64
	for i := 0; i < len(point1); i++ {
		sum += math.Pow(point1[i]-point2[i], 2)
	}
	return math.Sqrt(sum)
}

func KNN(k int, dataset []DataPoint, newPoint []float64) string {
	type Neighbor struct {
		Index    int
		Distance float64
	}

	var neighbors []Neighbor
	for i, data := range dataset {
		distance := EuclideanDistance(data.Features, newPoint)
		neighbors = append(neighbors, Neighbor{Index: i, Distance: distance})
	}

	for i := 0; i < len(neighbors)-1; i++ {
		for j := i + 1; j < len(neighbors); j++ {
			if neighbors[i].Distance > neighbors[j].Distance {
				neighbors[i], neighbors[j] = neighbors[j], neighbors[i]
			}
		}
	}

	votes := make(map[string]int)
	for i := 0; i < k; i++ {
		label := dataset[neighbors[i].Index].Label
		votes[label]++
	}

	var maxLabel string
	maxCount := 0
	for label, count := range votes {
		if count > maxCount {
			maxLabel = label
			maxCount = count
		}
	}

	return maxLabel
}

func main() {

	dataset, err := LoadDataset("dataset.csv")
	if err != nil {
		log.Fatal("Error loading dataset:", err)
	}

	newPoint := []float64{5.1, 3.5, 1.4, 0.2}

	k := 3
	prediction := KNN(k, dataset, newPoint)
	fmt.Println("Prediksi untuk titik data baru:", prediction)
}
